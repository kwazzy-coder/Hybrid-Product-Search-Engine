import os
import re
import asyncio
import io
import json
import time
import uuid
from difflib import get_close_matches
from datetime import datetime, timezone
from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pymongo import MongoClient
from apscheduler.schedulers.background import BackgroundScheduler

from typing import List, Optional
from pydantic import BaseModel, Field

from query_understanding import parse_query, QueryIntent
from retrieval.bm25_retriever import BM25Retriever
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.clip_retriever import CLIPRetriever
from retrieval.hybrid import reciprocal_rank_fusion
from reranker.features import build_feature_vector
from reranker.ltr_model import LTRModel
from analytics import SearchAnalytics
from ingestion import create_cross_platform_dataset
from deduplication import find_cross_source_duplicates, build_dedup_index, price_competitiveness_score, diversify_results
from feedback_db import log_click_event, init_db, log_impressions, create_alert, count_active_alerts
from paths import CATALOG_MODE, CATALOG_PATH, INDEX_DIR, MATCHES_PATH
from query_understanding import category_matches, KNOWN_CATEGORIES
from functools import partial
from pipeline import Filters, TEXT_WEIGHTS, fuse, rank, rank_photo, resolve_filters
from reranker.features import FEATURE_VERSION
from sizes import ONE_SIZE, normalize_size, size_sort_key
from collections import Counter
import traceback

app = FastAPI(title="Dhaaga search API", description="Hybrid search across Indian fashion brands")

# When deployed as one container, the built React app is served from the same URL.
FRONTEND_DIST = os.getenv("FRONTEND_DIST", "")
SERVE_FRONTEND = bool(FRONTEND_DIST) and os.path.isdir(FRONTEND_DIST)
LOG_IMPRESSIONS = os.getenv("LOG_IMPRESSIONS", "true").lower() == "true"
# "lite" = BM25 + re-ranker only (no MiniLM/CLIP, no torch): fits 512 MB free hosting tiers.
LITE_MODE = os.getenv("SEARCH_MODE", "full").lower() == "lite"
ALERTS_ENABLED = os.getenv("ALERTS_ENABLED", "true").lower() == "true"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

if not SERVE_FRONTEND:
    @app.get("/")
    async def root():
        return {
            "service": "Dhaaga search API",
            "status": "ok",
            "docs": "/docs",
            "search": "/api/search?q=red+dress",
        }


# Serve product images as static files
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Enable CORS for frontend and API Gateway
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = "ecommerce"
MOCK_SERVICES = os.getenv("MOCK_SERVICES", "false").lower() == "true"

# Initialize singletons
mongo_client = None
db = None
mock_products = []
catalog_by_id = {}
catalog_category_set = set()
analytics = SearchAnalytics()
index_metadata = {"version": "unavailable", "document_count": 0}


def set_catalog(products):
    """Replace the in-memory catalog and its id lookup."""
    global mock_products, catalog_by_id, catalog_category_set
    mock_products = products
    catalog_by_id = {p["_id"]: p for p in products}
    catalog_category_set = {str(p.get("category", "")).lower() for p in products if p.get("category")}


def load_local_catalog():
    """Load the catalog file chosen in paths.py: the crawled live catalog, or the demo one."""
    if mock_products:
        return mock_products
    if CATALOG_PATH and os.path.exists(CATALOG_PATH):
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            set_catalog(json.load(f))
        print(f"Loaded {len(mock_products)} products ({CATALOG_MODE} catalog) from {CATALOG_PATH}.")
    return mock_products


def inline_fallback_products():
    return [{
        "_id": f"prod_{i}", "name": f"Libas Women Red Kurta {i}", "category": "kurta",
        "price": 399, "mrp": 899, "discount_percent": 55, "rating": 4.2,
        "review_count": 120, "in_stock": True, "color": ["red"], "tags": ["kurta"],
        "embedding_text": f"Libas Women Red Kurta {i} kurta red"
    } for i in range(100)]


def hydrate(product_ids):
    """Fetch full product records for candidate ids. Returns {id: product}."""
    global MOCK_SERVICES
    if not MOCK_SERVICES:
        try:
            products = list(db["products"].find({"_id": {"$in": product_ids}}))
            if products:
                return {p["_id"]: p for p in products}
            # A fresh MongoDB volume has no products; the indexes refer to the local catalog.
        except Exception as e:
            print(f"MongoDB query failed: {e}. Falling back to the local catalog.")
            MOCK_SERVICES = True
    load_local_catalog()
    return {pid: catalog_by_id[pid] for pid in product_ids if pid in catalog_by_id}


def cross_platform_info(dedup_info, product_map):
    cheapest = catalog_by_id.get(dedup_info["cheapest_id"]) or product_map.get(dedup_info["cheapest_id"], {})
    return {
        "cheapest_price": dedup_info["cheapest_price"],
        "cheapest_source": cheapest.get("source_name") or cheapest.get("source", "unknown"),
        "cheapest_url": cheapest.get("source_url"),
        "available_on": dedup_info["group_size"],
        "price_rank": dedup_info["price_rank"],
    }


def catalog_products():
    """Return the active catalog for suggestions and demo ingestion."""
    if MOCK_SERVICES:
        return mock_products or load_local_catalog()
    return load_local_catalog()


def suggest_queries(raw_query: str, limit: int = 5) -> list[str]:
    query = raw_query.strip().lower()
    if len(query) < 2:
        return []
    terms = set()
    for product in catalog_products():
        terms.add(str(product.get("category", "")).lower())
        terms.update(str(tag).lower() for tag in product.get("tags", []))
        terms.update(str(color).lower() for color in product.get("color", []))
        terms.update(str(product.get("name", "")).lower().split())
    corrected = get_close_matches(query, terms, n=1, cutoff=0.72)
    prefix_matches = sorted(term for term in terms if term.startswith(query))[:limit]
    return list(dict.fromkeys(corrected + prefix_matches))[:limit]


def explain_result(intent, product, ltr_score: float, rrf_score: float) -> list[str]:
    reasons = []
    product_text = " ".join([
        str(product.get("category", "")), str(product.get("name", "")),
        " ".join(str(tag) for tag in product.get("tags", [])),
    ]).lower()
    if intent.category and intent.category.lower() in product_text:
        reasons.append("matched category")
    if intent.color and any(color.lower() in [str(value).lower() for value in product.get("color", [])] for color in intent.color):
        reasons.append("matched color")
    if intent.max_price is not None and product.get("price", 0) <= intent.max_price:
        reasons.append("within budget")
    if product.get("in_stock", True):
        reasons.append("in stock")
    reasons.append(f"hybrid relevance {rrf_score:.3f}")
    reasons.append(f"rerank score {float(ltr_score):.3f}")
    return reasons

if CATALOG_MODE == "live":
    # The crawled catalog is served from memory; MongoDB isn't needed for product data.
    MOCK_SERVICES = True
    print(f"Live catalog mode: {CATALOG_PATH} with indexes in {INDEX_DIR}")

if MOCK_SERVICES:
    print("Running in MOCK_SERVICES mode: products are served from the local catalog file.")
else:
    try:
        mongo_client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=2000)
        db = mongo_client[DB_NAME]
        mongo_client.admin.command('ping')
        print("Connected to MongoDB successfully.")
    except Exception as e:
        print(f"MongoDB connection failed: {e}. Automatically falling back to MOCK_SERVICES mode.")
        MOCK_SERVICES = True

if MOCK_SERVICES:
    load_local_catalog()
    if not mock_products:
        set_catalog(inline_fallback_products())
        print(f"No catalog file found; generated {len(mock_products)} inline fallback products.")

bm25 = BM25Retriever()
faiss = FAISSRetriever()
clip = CLIPRetriever()
ltr_model = LTRModel()

# Cross-platform deduplication index (product_id -> dedup info)
dedup_index = {}

# Flag to trigger incremental index rebuilds
bm25_dirty = False

def rebuild_indices_from_db():
    global bm25_dirty, index_metadata
    print("Rebuilding search indices...")
    try:
        if MOCK_SERVICES:
            products = mock_products
        else:
            products = list(db["products"].find({}))
            
        if not products:
            print("No products available to build indices.")
            return False
            
        # Rebuild BM25
        bm25.build(products)
        bm25.save()
        
        # Rebuild FAISS (skipped in lite mode: no embedding model there)
        if not LITE_MODE:
            faiss.index = FAISSRetriever().index  # Reset index
            faiss.id_map = {}
            faiss.add_products(products)
            faiss.save()

        index_metadata = {
            **index_metadata,
            "version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "catalog_mode": CATALOG_MODE,
            "document_count": len(products),
            "rebuilt_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path = os.path.join(INDEX_DIR, "index_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(index_metadata, metadata_file, indent=2)
        
        bm25_dirty = False
        print("Indices successfully rebuilt and saved.")
        return True
    except Exception as e:
        print(f"Error building indices: {e}")
        return False

# Setup background scheduler for BM25 (rebuild every 5 mins if dirty)
scheduler = BackgroundScheduler(timezone="UTC")  # explicit, so odd TZ settings can't crash startup

def check_and_rebuild_indices():
    global bm25_dirty
    if bm25_dirty:
        print("Dirty flag detected. Triggering scheduled index rebuild...")
        rebuild_indices_from_db()

scheduler.add_job(check_and_rebuild_indices, 'interval', minutes=5)

@app.on_event("startup")
async def startup_event():
    global dedup_index
    # Initialize SQLite feedback DB
    init_db()
    # Start scheduler
    scheduler.start()
    
    # Try loading existing indices
    bm25_loaded = bm25.load()
    faiss_loaded = True if LITE_MODE else faiss.load()
    clip_loaded = False if LITE_MODE else clip.load()
    if LITE_MODE:
        print("SEARCH_MODE=lite: BM25 + re-ranking only (vector and photo search are off).")
    global index_metadata
    metadata_path = os.path.join(INDEX_DIR, "index_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            index_metadata = json.load(metadata_file)
    else:
        index_metadata = {"version": "legacy", "document_count": len(bm25.product_ids)}
    
    if not clip_loaded:
        print("CLIP index not found. Visual search will be disabled.")

    # Load the embedding models now rather than on the first search, so the first
    # searches (and photo search) don't race a half-loaded model. Takes a few seconds.
    if not LITE_MODE:
        started = time.perf_counter()
        if faiss.index.ntotal > 0:
            faiss._load_model()
        if clip.index.ntotal > 0:
            clip._load_model()
        print(f"Embedding models ready in {time.perf_counter() - started:.1f}s "
              f"(MiniLM: {'ok' if faiss.model is not None else 'unavailable'}, "
              f"CLIP: {'ok' if clip.model is not None else 'unavailable'})")
    
    if not bm25_loaded or not faiss_loaded:
        print("Indices not found or incomplete. Rebuilding from DB...")
        rebuild_indices_from_db()
        # Reload models
        ltr_model.load()
    
    # Build cross-source deduplication index
    if CATALOG_MODE == "live":
        if os.path.exists(MATCHES_PATH):
            with open(MATCHES_PATH, "r", encoding="utf-8") as fh:
                groups = json.load(fh)
            dedup_index = build_dedup_index(mock_products, groups)
            print(f"Cross-store matches loaded: {len(groups)} products sold by more than one store")
    elif MOCK_SERVICES and mock_products:
        try:
            dup_groups = find_cross_source_duplicates(clip, mock_products)
            dedup_index = build_dedup_index(mock_products, dup_groups)
            print(f"Cross-source dedup index built: {len(dedup_index)} products have cross-platform duplicates")
        except Exception as e:
            print(f"Dedup index build failed (non-critical): {e}")

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()
    if mongo_client:
        mongo_client.close()

# ─────────────────────────────────────────────────────────────── search
def open_upload(contents: bytes, filename: Optional[str], content_type: Optional[str]):
    """Open an uploaded photo: upright (phone EXIF rotation), RGB, HEIC supported when pillow-heif is installed."""
    from PIL import Image, ImageOps
    name = (filename or "").lower()
    ctype = (content_type or "").lower()
    if ctype and not ctype.startswith("image/") and ctype != "application/octet-stream":
        raise HTTPException(status_code=400, detail="File uploaded is not an image")
    if ctype in ("image/heic", "image/heif") or name.endswith((".heic", ".heif")):
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            raise HTTPException(status_code=400, detail="iPhone HEIC photos need `pip install pillow-heif` on the server. "
                                                        "Or upload a JPG or PNG instead.")
    try:
        image = Image.open(io.BytesIO(contents))
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Couldn't read that image ({type(e).__name__}). Try a JPG or PNG.")


def known_categories() -> set:
    return KNOWN_CATEGORIES | catalog_category_set


async def parse_intent(text: str) -> QueryIntent:
    try:
        return await asyncio.to_thread(parse_query, text)
    except Exception as e:
        print(f"Query parsing failed: {e}")
        return QueryIntent(original_query=text, expanded_terms=[text])


async def run_retrievers(calls):
    """Run retrievers in parallel; if one fails, keep the others' results."""
    outcomes = await asyncio.gather(*(asyncio.to_thread(fn, arg) for _, fn, arg in calls), return_exceptions=True)
    results = []
    for (name, _, _), outcome in zip(calls, outcomes):
        if isinstance(outcome, BaseException):
            print(f"{name} retrieval failed: {outcome!r}")
            results.append([])
        else:
            results.append(outcome)
    return results


def build_response(*, query_id, query_text, intent, ranked, product_map, fused, page, limit, started_at,
                   mode, background_tasks: BackgroundTasks, notice=None, score_kind="ltr"):
    fused_scores = dict(fused)
    total = len(ranked)
    start = (page - 1) * limit
    page_items = ranked[start:start + limit]
    results = []
    for offset, cand in enumerate(page_items):
        product = product_map[cand.product_id]
        enriched = product.copy()
        enriched["_score"] = float(cand.score)
        enriched["_score_kind"] = score_kind
        if score_kind == "visual":
            enriched["_similarity"] = round(float(cand.fused_score), 4)
            enriched["_explanation"] = [f"looks {cand.fused_score:.0%} similar to your photo"]
        else:
            enriched["_explanation"] = explain_result(intent, product, cand.score, fused_scores.get(cand.product_id, 0.0))
        enriched["_query_id"] = query_id
        enriched["_position"] = start + offset
        enriched["source"] = product.get("source", "unknown")
        dedup_info = dedup_index.get(cand.product_id)
        if dedup_info:
            enriched["_cross_platform"] = cross_platform_info(dedup_info, product_map)
        results.append(enriched)

    latency_ms = (time.perf_counter() - started_at) * 1000
    analytics.record_search(query_id, query_text, latency_ms, total)
    if LOG_IMPRESSIONS and page_items:
        # Record exactly what was shown, with the features used to rank it (for retraining).
        background_tasks.add_task(
            log_impressions, query_id, query_text, mode,
            [(start + i, c.product_id, c.features) for i, c in enumerate(page_items)],
            FEATURE_VERSION, ltr_model.version,
        )
    response = {
        "query_id": query_id,
        "query": query_text,
        "intent": intent.model_dump(),
        "total": total,
        "latency_ms": round(latency_ms, 2),
        "results": results,
    }
    if notice:
        response["notice"] = notice
    return response


@app.get("/search")
@app.get("/api/search")
async def search(
    background_tasks: BackgroundTasks,
    q: str = Query(..., description="Raw search query"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=50, description="Items per page"),
    category: Optional[str] = Query(None, description="Category filter override"),
    min_price: Optional[float] = Query(None, description="Min price filter override"),
    max_price: Optional[float] = Query(None, description="Max price filter override"),
    min_rating: Optional[float] = Query(None, description="Min rating filter override"),
    in_stock_only: bool = Query(True, description="Filter for in-stock items"),
    size: Optional[str] = Query(None, description="Only items in stock in this size (e.g. M, XL, 32, 8)"),
):
    try:
        started_at = time.perf_counter()
        query_id = str(uuid.uuid4())
        intent = await parse_intent(q)

        if LITE_MODE:
            (bm25_results,) = await run_retrievers([("BM25", bm25.search, intent.expanded_terms)])
            faiss_results, clip_results = [], []
        else:
            bm25_results, faiss_results, clip_results = await run_retrievers([
                ("BM25", bm25.search, intent.expanded_terms),
                ("FAISS", faiss.search, intent.original_query),
                ("CLIP", clip.search, intent.original_query),
            ])
        fused = fuse(bm25_results, faiss_results, clip_results, weights=TEXT_WEIGHTS)
        filters = resolve_filters(intent, Filters(category, min_price, max_price, min_rating, in_stock_only, size),
                                  known_categories())
        product_map = hydrate([pid for pid, _ in fused]) if fused else {}
        ranked = rank(intent, fused, product_map, dict(clip_results), ltr_model, dedup_index, filters)
        return build_response(query_id=query_id, query_text=q, intent=intent, ranked=ranked, product_map=product_map,
                              fused=fused, page=page, limit=limit, started_at=started_at, mode="text",
                              background_tasks=background_tasks)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Search endpoint error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/search/image")
@app.post("/api/search/image")
async def search_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    q: Optional[str] = Form(None),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=50, description="Items per page"),
    category: Optional[str] = Query(None, description="Category filter override"),
    min_price: Optional[float] = Query(None, description="Min price filter override"),
    max_price: Optional[float] = Query(None, description="Max price filter override"),
    min_rating: Optional[float] = Query(None, description="Min rating filter override"),
    in_stock_only: bool = Query(True, description="Filter for in-stock items"),
    size: Optional[str] = Query(None, description="Only items in stock in this size"),
):
    try:
        started_at = time.perf_counter()
        query_id = str(uuid.uuid4())

        if LITE_MODE:
            raise HTTPException(status_code=503, detail="Photo search is turned off on this server (lite mode).")
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image is larger than 10 MB")
        image = open_upload(contents, file.filename, file.content_type)

        text = (q or "").strip()
        query_text = text or "[photo]"
        intent = await parse_intent(text) if text else QueryIntent(original_query="[photo]", expanded_terms=[])

        # The photo finds candidates; text (if any) only filters and nudges them. See pipeline.rank_photo.
        (visual,) = await run_retrievers([("CLIP", partial(clip.search_by_image, top_k=200), image)])

        notice = None
        if not visual:
            if clip.index.ntotal == 0:
                notice = "Photo search needs the CLIP image index: install torch and run scripts/build_live_index.py."
            elif clip.model is None:
                reason = clip.load_error or "unknown error (see the API log)"
                notice = f"Photo search is unavailable: the CLIP model failed to load on the server ({reason})."

        filters = resolve_filters(intent, Filters(category, min_price, max_price, min_rating, in_stock_only, size),
                                  known_categories())
        product_map = hydrate([pid for pid, _ in visual]) if visual else {}
        ranked = rank_photo(intent, visual, product_map, dedup_index, filters, has_text=bool(text))
        return build_response(query_id=query_id, query_text=query_text, intent=intent, ranked=ranked,
                              product_map=product_map, fused=visual, page=page, limit=limit, started_at=started_at,
                              mode="image+text" if text else "image", background_tasks=background_tasks,
                              notice=notice, score_kind="visual")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Image search endpoint error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")


@app.post("/admin/rebuild_indices")
def admin_rebuild():
    success = rebuild_indices_from_db()
    if not success:
        raise HTTPException(status_code=500, detail="Rebuild failed")
    return {"status": "success", "message": "Indices successfully rebuilt"}

@app.post("/admin/upsert_product")
def admin_upsert_product(product: dict):
    global bm25_dirty
    try:
        faiss.add_products([product])
        faiss.save()
        bm25_dirty = True
        return {"status": "success", "message": f"Product {product.get('_id')} upserted to FAISS index successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/delete_product")
def admin_delete_product(payload: dict):
    global bm25_dirty
    product_id = payload.get("_id")
    if not product_id:
        raise HTTPException(status_code=400, detail="Missing product _id")
    try:
        faiss.exclusion_list.add(product_id)
        faiss.save()
        bm25_dirty = True
        return {"status": "success", "message": f"Product {product_id} marked as deleted in FAISS"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/reload_ltr")
def admin_reload_ltr():
    # Pick up a model retrained on live clicks (models_live/ltr_model.lgb) if one now exists.
    from reranker.ltr_model import default_model_path
    ltr_model.model_path = default_model_path()
    success = ltr_model.load()
    if not success:
        raise HTTPException(status_code=500, detail="LTR load failed")
    return {"status": "success", "message": "LTR model reloaded"}

@app.post("/admin/ingest")
def admin_ingest_products(payload: dict):
    """Ingest a catalog batch, then atomically rebuild the retrieval indexes."""
    products = payload.get("products", [])
    if not isinstance(products, list) or not products:
        raise HTTPException(status_code=400, detail="Payload must include a non-empty products list")
    required_fields = {"_id", "name", "category", "price"}
    invalid = [product.get("_id", "unknown") for product in products if not required_fields.issubset(product)]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid products: {invalid[:5]}")
    global mock_products, MOCK_SERVICES
    if MOCK_SERVICES or db is None:
        existing = {product["_id"]: product for product in catalog_products()}
        existing.update({product["_id"]: product for product in products})
        set_catalog(list(existing.values()))
        MOCK_SERVICES = True
    else:
        for product in products:
            db["products"].replace_one({"_id": product["_id"]}, product, upsert=True)
    if not rebuild_indices_from_db():
        raise HTTPException(status_code=500, detail="Products stored but index rebuild failed")
    return {"status": "success", "ingested": len(products), "index": index_metadata}

@app.get("/suggestions")
@app.get("/api/suggestions")
def suggestions(q: str = Query(..., min_length=2), limit: int = Query(5, ge=1, le=10)):
    return {"query": q, "suggestions": suggest_queries(q, limit)}


@app.post("/analytics/click")
@app.post("/api/analytics/click")
def record_click(payload: dict):
    query_id = payload.get("query_id")
    product_id = payload.get("product_id")
    if not query_id or not product_id:
        raise HTTPException(status_code=400, detail="query_id and product_id are required")
    analytics.record_click(query_id, product_id)
    return {"status": "recorded"}


@app.get("/analytics/summary")
@app.get("/api/analytics/summary")
def analytics_summary():
    return analytics.summary()


@app.get("/stats")
@app.get("/api/stats")
def stats():
    sources = set()
    catalog = mock_products if MOCK_SERVICES else []
    if catalog:
        sources = set(p.get("source", "unknown") for p in catalog)
    store_counts = Counter((p.get("source", "unknown"), p.get("source_name") or p.get("source", "unknown")) for p in catalog)
    category_counts = Counter(p.get("category") for p in catalog if p.get("category"))
    size_counts = Counter(
        normalize_size(s) for p in catalog for s in (p.get("available_sizes") or [])
    )
    size_counts.pop(ONE_SIZE, None)
    size_counts.pop(None, None)
    top_sizes = sorted((s for s, _ in size_counts.most_common(16)), key=size_sort_key)
    return {
        "catalog": {
            "mode": CATALOG_MODE,
            "products": len(catalog),
            "stores": [{"id": sid, "name": name, "count": n} for (sid, name), n in store_counts.most_common()],
            "top_categories": [{"name": c, "count": n} for c, n in category_counts.most_common(12)],
            "top_sizes": top_sizes,
            "has_ratings": any((p.get("review_count") or 0) > 0 for p in catalog),
            "last_scraped_at": max((p.get("scraped_at", "") for p in catalog), default="") or None,
        },
        "features": {
            "visual_search_ready": clip.index.ntotal > 0 and not LITE_MODE and clip.load_error is None,
            "visual_search_error": ("lite mode" if LITE_MODE else clip.load_error
                                    or (None if clip.index.ntotal > 0 else "no CLIP image index (run build_live_index.py)")),
            "search_mode": "lite" if LITE_MODE else "full",
            "alerts_enabled": ALERTS_ENABLED,
            "logging_impressions": LOG_IMPRESSIONS,
        },
        "total_products_indexed": len(bm25.product_ids),
        "faiss_index_size": faiss.index.ntotal,
        "clip_index_size": clip.index.ntotal,
        "clip_model_loaded": clip.model is not None,
        "clip_load_error": clip.load_error,
        "minilm_loaded": faiss.model is not None,
        "minilm_load_error": faiss.load_error,
        "faiss_exclusion_list_size": len(faiss.exclusion_list),
        "bm25_dirty": bm25_dirty,
        "ltr_model_loaded": ltr_model.model is not None,
        "ltr_model_version": ltr_model.version,
        "cross_platform": {
            "sources": sorted(sources),
            "dedup_entries": len(dedup_index),
        },
        "index": index_metadata
    }


class FeedbackPayload(BaseModel):
    query: str
    clicked_id: str
    shown_results: List[str] = Field(default_factory=list)
    query_id: Optional[str] = None
    position: Optional[int] = None
    timestamp: Optional[str] = None


@app.post("/feedback")
@app.post("/api/feedback")
def record_feedback(payload: FeedbackPayload):
    if not payload.query or not payload.clicked_id:
        raise HTTPException(status_code=400, detail="query and clicked_id are required")
    event_id = log_click_event(
        query=payload.query,
        shown_results=payload.shown_results,
        clicked_id=payload.clicked_id,
        timestamp=payload.timestamp,
        query_id=payload.query_id,
        position=payload.position,
    )
    if payload.query_id:
        analytics.record_click(payload.query_id, payload.clicked_id)
    return {"status": "success", "event_id": event_id}


class AlertPayload(BaseModel):
    product_id: str
    target_price: float = Field(gt=0)
    email: str


@app.post("/api/alerts")
def create_price_alert(payload: AlertPayload):
    """Ask to be emailed when a product drops to target_price or below (checked after each crawl)."""
    if not ALERTS_ENABLED:
        raise HTTPException(status_code=403, detail="Price alerts are turned off on this server")
    product = catalog_by_id.get(payload.product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    email = payload.email.strip()
    if not EMAIL_RE.match(email) or len(email) > 254:
        raise HTTPException(status_code=400, detail="Please enter a valid email address")
    current = float(product.get("price") or 0)
    if payload.target_price >= current:
        raise HTTPException(status_code=400, detail=f"It's already ₹{current:,.0f}. Pick a price below that.")
    if count_active_alerts(email) >= 20:
        raise HTTPException(status_code=429, detail="That email already has 20 active alerts")
    alert_id = create_alert(product, email, payload.target_price)
    return {
        "status": "created",
        "alert_id": alert_id,
        "message": f"We'll email {email} when it drops to ₹{payload.target_price:,.0f} or less.",
    }


# Serve the built frontend last, so every API route above takes precedence.
if SERVE_FRONTEND:
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
