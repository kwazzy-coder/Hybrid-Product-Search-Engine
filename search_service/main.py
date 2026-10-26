import os
import asyncio
import json
import time
import uuid
from difflib import get_close_matches
from datetime import datetime, timezone
from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pymongo import MongoClient
from apscheduler.schedulers.background import BackgroundScheduler

from typing import List, Optional
from pydantic import BaseModel, Field

from query_understanding import parse_query
from retrieval.bm25_retriever import BM25Retriever
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.clip_retriever import CLIPRetriever
from retrieval.hybrid import reciprocal_rank_fusion
from reranker.features import build_feature_vector
from reranker.ltr_model import LTRModel
from analytics import SearchAnalytics
from ingestion import create_cross_platform_dataset
from deduplication import find_cross_source_duplicates, build_dedup_index, price_competitiveness_score, diversify_results
from feedback_db import log_click_event, init_db
from paths import CATALOG_MODE, CATALOG_PATH, INDEX_DIR, MATCHES_PATH
from query_understanding import category_matches, KNOWN_CATEGORIES
from collections import Counter
import traceback

app = FastAPI(title="Hybrid Product Search Engine API")


@app.get("/")
async def root():
    return {
        "service": "Hybrid Product Search Engine API",
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


def resolve_category_filter(explicit, intent):
    """UI filters always apply; a parsed category only filters if it's one we actually know."""
    if explicit:
        return explicit.lower().strip()
    guessed = (intent.category or "").lower().strip()
    if guessed and (guessed in KNOWN_CATEGORIES or guessed in catalog_category_set):
        return guessed
    return None


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
        
        # Rebuild FAISS
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
    faiss_loaded = faiss.load()
    clip_loaded = clip.load()
    global index_metadata
    metadata_path = os.path.join(INDEX_DIR, "index_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            index_metadata = json.load(metadata_file)
    else:
        index_metadata = {"version": "legacy", "document_count": len(bm25.product_ids)}
    
    if not clip_loaded:
        print("CLIP index not found. Visual search will be disabled.")
    
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

@app.get("/search")
async def search(
    q: str = Query(..., description="Raw search query"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=50, description="Items per page"),
    category: str = Query(None, description="Category filter override"),
    min_price: float = Query(None, description="Min price filter override"),
    max_price: float = Query(None, description="Max price filter override"),
    min_rating: float = Query(None, description="Min rating filter override"),
    in_stock_only: bool = Query(True, description="Filter for in-stock items")
):
    try:
        started_at = time.perf_counter()
        query_id = str(uuid.uuid4())
        global MOCK_SERVICES, mock_products
        # Step 1: parse query intent
        try:
            intent = await asyncio.to_thread(parse_query, q)
        except Exception as e:
            print(f"Query parsing failed: {e}")
            # Build dummy intent
            from query_understanding import QueryIntent
            intent = QueryIntent(original_query=q, expanded_terms=[q])

        # Step 2: retrieve (parallel using asyncio.to_thread — 3-way)
        try:
            bm25_results, faiss_results, clip_results = await asyncio.gather(
                asyncio.to_thread(bm25.search, intent.expanded_terms),
                asyncio.to_thread(faiss.search, intent.original_query),
                asyncio.to_thread(clip.search, intent.original_query)
            )
        except Exception as e:
            print(f"Retrieval failed: {e}")
            bm25_results, faiss_results, clip_results = [], [], []

        # Step 3: fuse (3-way weighted RRF)
        merged = reciprocal_rank_fusion(
            bm25_results, faiss_results, clip_results,
            weights=[1.0, 1.0, 0.8]
        )
        
        # Build CLIP score lookup for re-ranking features
        clip_score_map = {pid: score for pid, score in clip_results}
        if not merged:
            analytics.record_search(query_id, q, (time.perf_counter() - started_at) * 1000, 0)
            return {
                "query_id": query_id,
                "query": q,
                "intent": intent.model_dump(),
                "total": 0,
                "results": []
            }

        # Step 4: fetch product details for top 100 merged candidates
        product_ids = [pid for pid, _ in merged]
        product_map = hydrate(product_ids)

        # Step 5: rerank
        feature_vectors = []
        candidates_to_rank = []
        
        for pid, score in merged:
            if pid in product_map:
                product = product_map[pid]
                # Use intent max_price and min_rating but let explicit UI overrides take precedence
                query_max_price = max_price if max_price is not None else intent.max_price
                query_min_rating = min_rating if min_rating is not None else intent.min_rating
                
                # Temporary intent copy with overrides for feature calculation
                overridden_intent = intent.model_copy(update={
                    "max_price": query_max_price,
                    "min_rating": query_min_rating
                })
                
                product_clip_score = clip_score_map.get(pid, 0.0)
                
                # Cross-platform price competitiveness (f12)
                dedup_info = dedup_index.get(pid)
                pc_score = price_competitiveness_score(product, dedup_info) if dedup_info else 0.5
                
                fv = build_feature_vector(overridden_intent, product, score, clip_score=product_clip_score, price_competitiveness=pc_score)
                feature_vectors.append(fv)
                candidates_to_rank.append(pid)

        if feature_vectors:
            try:
                ltr_scores = ltr_model.predict(feature_vectors)
            except Exception as e:
                print(f"LTR prediction error: {e}")
                ltr_scores = [fv[0] for fv in feature_vectors] # fallback to RRF score
        else:
            ltr_scores = []

        # Step 6: sort + source diversity
        ranked = sorted(zip(candidates_to_rank, ltr_scores), key=lambda x: x[1], reverse=True)
        
        # Apply source-diversity constraint (prevents one platform from dominating results)
        ranked = diversify_results(ranked, product_map, lambda_diversity=0.3)
        
        rrf_score_map = dict(merged)

        # Step 7: Apply post-filters
        filtered_results = []
        
        # Active filters (UI overrides or LLM extracted intent)
        filter_category = resolve_category_filter(category, intent)
        filter_max_price = max_price if max_price is not None else intent.max_price
        filter_min_price = min_price
        filter_min_rating = min_rating if min_rating is not None else intent.min_rating

        for pid, ltr_score in ranked:
            product = product_map[pid]
            
            # Category match check
            if filter_category and not category_matches(product, filter_category):
                continue
                
            # Price check
            price = product.get("price", 0)
            if filter_max_price is not None and price > filter_max_price:
                continue
            if filter_min_price is not None and price < filter_min_price:
                continue
                
            # Rating check
            rating = product.get("rating") or 0.0
            if filter_min_rating is not None and rating < filter_min_rating:
                continue
                
            # Stock check
            if in_stock_only and not product.get("in_stock", True):
                continue
                
            # Enrich product dict with score for transparency
            enriched_product = product.copy()
            enriched_product["_score"] = float(ltr_score)
            enriched_product["_explanation"] = explain_result(
                intent, product, ltr_score, rrf_score_map.get(pid, 0.0)
            )
            
            # Add cross-platform info
            enriched_product["source"] = product.get("source", "unknown")
            dedup_info = dedup_index.get(pid)
            if dedup_info:
                enriched_product["_cross_platform"] = cross_platform_info(dedup_info, product_map)
            
            filtered_results.append(enriched_product)

        # Step 8: paginate
        total_results = len(filtered_results)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        page_results = filtered_results[start_idx:end_idx]

        latency_ms = (time.perf_counter() - started_at) * 1000
        analytics.record_search(query_id, q, latency_ms, total_results)
        return {
            "query_id": query_id,
            "query": q,
            "intent": intent.model_dump(),
            "total": total_results,
            "latency_ms": round(latency_ms, 2),
            "results": page_results
        }
    except Exception as e:
        print(f"Search endpoint error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search/image")
async def search_image(
    file: UploadFile = File(...),
    q: Optional[str] = Form(None),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=50, description="Items per page"),
    category: Optional[str] = Query(None, description="Category filter override"),
    min_price: Optional[float] = Query(None, description="Min price filter override"),
    max_price: Optional[float] = Query(None, description="Max price filter override"),
    min_rating: Optional[float] = Query(None, description="Min rating filter override"),
    in_stock_only: bool = Query(True, description="Filter for in-stock items")
):
    try:
        started_at = time.perf_counter()
        query_id = str(uuid.uuid4())
        global MOCK_SERVICES, mock_products

        # Step 1: Validate uploaded image
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File uploaded is not an image")
            
        contents = await file.read()
        from PIL import Image
        import io
        try:
            image = Image.open(io.BytesIO(contents))
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        # Step 2: Extract text intent if optional text prompt provided
        intent_query_text = q.strip() if (q and q.strip()) else "[Uploaded Image]"
        if q and q.strip():
            try:
                intent = await asyncio.to_thread(parse_query, q.strip())
            except Exception as e:
                print(f"Query parsing failed for image search text prompt: {e}")
                from query_understanding import QueryIntent
                intent = QueryIntent(original_query=q.strip(), expanded_terms=[q.strip()])
        else:
            from query_understanding import QueryIntent
            intent = QueryIntent(original_query="[Uploaded Image]", expanded_terms=[])

        # Step 3: Run image encoding & retrieval
        if q and q.strip():
            # Multimodal: Image + Text query
            bm25_results, faiss_results, clip_results = await asyncio.gather(
                asyncio.to_thread(bm25.search, intent.expanded_terms),
                asyncio.to_thread(faiss.search, intent.original_query),
                asyncio.to_thread(clip.search_by_image, image)
            )
            merged = reciprocal_rank_fusion(
                bm25_results, faiss_results, clip_results,
                weights=[0.8, 0.8, 1.2]
            )
        else:
            # Pure Visual Image Search
            clip_results = await asyncio.to_thread(clip.search_by_image, image)
            merged = clip_results

        clip_score_map = {pid: score for pid, score in clip_results}
        if not merged:
            analytics.record_search(query_id, intent_query_text, (time.perf_counter() - started_at) * 1000, 0)
            return {
                "query_id": query_id,
                "query": intent_query_text,
                "intent": intent.model_dump(),
                "total": 0,
                "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "results": []
            }

        # Step 4: Fetch product candidates
        product_ids = [pid for pid, _ in merged]
        product_map = hydrate(product_ids)

        # Step 5: LTR Re-ranking
        feature_vectors = []
        candidates_to_rank = []

        for pid, score in merged:
            if pid in product_map:
                product = product_map[pid]
                query_max_price = max_price if max_price is not None else intent.max_price
                query_min_rating = min_rating if min_rating is not None else intent.min_rating

                overridden_intent = intent.model_copy(update={
                    "max_price": query_max_price,
                    "min_rating": query_min_rating
                })

                product_clip_score = clip_score_map.get(pid, score)
                dedup_info = dedup_index.get(pid)
                pc_score = price_competitiveness_score(product, dedup_info) if dedup_info else 0.5

                fv = build_feature_vector(overridden_intent, product, score, clip_score=product_clip_score, price_competitiveness=pc_score)
                feature_vectors.append(fv)
                candidates_to_rank.append(pid)

        if feature_vectors and ltr_model.model:
            try:
                ltr_scores = ltr_model.predict(feature_vectors)
            except Exception as e:
                print(f"LTR prediction error: {e}")
                ltr_scores = [fv[0] for fv in feature_vectors]
        else:
            ltr_scores = [fv[0] for fv in feature_vectors] if feature_vectors else []

        ranked = sorted(zip(candidates_to_rank, ltr_scores), key=lambda x: x[1], reverse=True)
        ranked = diversify_results(ranked, product_map, lambda_diversity=0.3)
        rrf_score_map = dict(merged)

        # Step 6: Post-filtering & Response enrichment
        filtered_results = []
        filter_category = resolve_category_filter(category, intent)
        filter_min_price = min_price if min_price is not None else getattr(intent, 'min_price', None)
        filter_max_price = max_price if max_price is not None else getattr(intent, 'max_price', None)
        filter_min_rating = min_rating if min_rating is not None else getattr(intent, 'min_rating', None)

        for pid, ltr_score in ranked:
            product = product_map[pid]

            if filter_category and not category_matches(product, filter_category):
                continue

            price = product.get("price", 0)
            if filter_max_price is not None and price > filter_max_price:
                continue
            if filter_min_price is not None and price < filter_min_price:
                continue

            rating = product.get("rating") or 0.0
            if filter_min_rating is not None and rating < filter_min_rating:
                continue

            if in_stock_only and not product.get("in_stock", True):
                continue

            enriched_product = product.copy()
            enriched_product["_score"] = float(ltr_score)
            enriched_product["_explanation"] = explain_result(
                intent, product, ltr_score, rrf_score_map.get(pid, 0.0)
            )

            enriched_product["source"] = product.get("source", "unknown")
            dedup_info = dedup_index.get(pid)
            if dedup_info:
                enriched_product["_cross_platform"] = cross_platform_info(dedup_info, product_map)

            filtered_results.append(enriched_product)

        # Step 7: Paginate
        total_results = len(filtered_results)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        page_results = filtered_results[start_idx:end_idx]

        latency_ms = (time.perf_counter() - started_at) * 1000
        analytics.record_search(query_id, intent_query_text, latency_ms, total_results)
        return {
            "query_id": query_id,
            "query": intent_query_text,
            "intent": intent.model_dump(),
            "total": total_results,
            "latency_ms": round(latency_ms, 2),
            "results": page_results
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Image search endpoint error: {e}")
        import traceback
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
def suggestions(q: str = Query(..., min_length=2), limit: int = Query(5, ge=1, le=10)):
    return {"query": q, "suggestions": suggest_queries(q, limit)}

@app.post("/analytics/click")
def record_click(payload: dict):
    query_id = payload.get("query_id")
    product_id = payload.get("product_id")
    if not query_id or not product_id:
        raise HTTPException(status_code=400, detail="query_id and product_id are required")
    analytics.record_click(query_id, product_id)
    return {"status": "recorded"}

@app.get("/analytics/summary")
def analytics_summary():
    return analytics.summary()

@app.get("/stats")
def stats():
    sources = set()
    catalog = mock_products if MOCK_SERVICES else []
    if catalog:
        sources = set(p.get("source", "unknown") for p in catalog)
    store_counts = Counter((p.get("source", "unknown"), p.get("source_name") or p.get("source", "unknown")) for p in catalog)
    category_counts = Counter(p.get("category") for p in catalog if p.get("category"))
    return {
        "catalog": {
            "mode": CATALOG_MODE,
            "products": len(catalog),
            "stores": [{"id": sid, "name": name, "count": n} for (sid, name), n in store_counts.most_common()],
            "top_categories": [{"name": c, "count": n} for c, n in category_counts.most_common(12)],
            "last_scraped_at": max((p.get("scraped_at", "") for p in catalog), default="") or None,
        },
        "total_products_indexed": len(bm25.product_ids),
        "faiss_index_size": faiss.index.ntotal,
        "clip_index_size": clip.index.ntotal,
        "clip_model_loaded": clip.model is not None,
        "faiss_exclusion_list_size": len(faiss.exclusion_list),
        "bm25_dirty": bm25_dirty,
        "ltr_model_loaded": ltr_model.model is not None,
        "cross_platform": {
            "sources": sorted(sources),
            "dedup_entries": len(dedup_index),
        },
        "index": index_metadata
    }

# ── Route aliases for direct frontend access (no gateway needed) ──
@app.get("/api/search")
async def api_search_alias(
    q: str = Query(..., description="Raw search query"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=50),
    category: str = Query(None),
    min_price: float = Query(None),
    max_price: float = Query(None),
    min_rating: float = Query(None),
    in_stock_only: bool = Query(True)
):
    return await search(q, page, limit, category, min_price, max_price, min_rating, in_stock_only)

@app.get("/api/stats")
def api_stats_alias():
    return stats()

@app.get("/api/suggestions")
def api_suggestions_alias(q: str = Query(..., min_length=2), limit: int = Query(5, ge=1, le=10)):
    return suggestions(q, limit)

@app.get("/api/analytics/summary")
def api_analytics_summary_alias():
    return analytics_summary()

class FeedbackPayload(BaseModel):
    query: str
    shown_results: List[str]
    clicked_id: str
    timestamp: Optional[str] = None

@app.post("/feedback")
def record_feedback(payload: FeedbackPayload):
    if not payload.query or not payload.clicked_id or not payload.shown_results:
        raise HTTPException(status_code=400, detail="query, clicked_id, and shown_results are required")
    event_id = log_click_event(
        query=payload.query,
        shown_results=payload.shown_results,
        clicked_id=payload.clicked_id,
        timestamp=payload.timestamp
    )
    return {"status": "success", "event_id": event_id, "message": "Click event logged to SQLite table click_events"}

@app.post("/api/analytics/click")
def api_record_click_alias(payload: dict):
    return record_click(payload)

@app.post("/api/feedback")
def api_record_feedback_alias(payload: FeedbackPayload):
    return record_feedback(payload)

@app.post("/api/search/image")
async def api_search_image_alias(
    file: UploadFile = File(...),
    q: Optional[str] = Form(None),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=50),
    category: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    min_rating: Optional[float] = Query(None),
    in_stock_only: bool = Query(True)
):
    return await search_image(file, q, page, limit, category, min_price, max_price, min_rating, in_stock_only)


