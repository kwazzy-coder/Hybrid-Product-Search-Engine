import os
import asyncio
import json
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pymongo import MongoClient
from apscheduler.schedulers.background import BackgroundScheduler

from query_understanding import parse_query
from retrieval.bm25_retriever import BM25Retriever
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.clip_retriever import CLIPRetriever
from retrieval.hybrid import reciprocal_rank_fusion
from reranker.features import build_feature_vector
from reranker.ltr_model import LTRModel

app = FastAPI(title="Hybrid Product Search Engine API")

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

def load_local_catalog():
    """Load the bundled catalog when MongoDB has not been seeded yet."""
    global mock_products
    if mock_products:
        return mock_products

    service_dir = os.path.dirname(os.path.abspath(__file__))
    for catalog_dir in ("/data", os.path.join(service_dir, "..", "data")):
        for filename in ("products_2k.json", "products_100k_backup.json"):
            path = os.path.join(catalog_dir, filename)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    mock_products = json.load(f)
                # The checked-in indexes are built from the 2K catalog.
                mock_products = mock_products[:2000]
                print(f"Loaded {len(mock_products)} products from local catalog: {filename}.")
                return mock_products

    return mock_products

if MOCK_SERVICES:
    print("Running in MOCK_SERVICES mode. Databases are mocked.")
    backup_path = None
    for p in ["../data/products_2k.json", "data/products_2k.json", "../data/products_100k_backup.json", "data/products_100k_backup.json"]:
        if os.path.exists(p):
            backup_path = p
            break
        
    if backup_path:
        with open(backup_path, "r") as f:
            mock_products = json.load(f)
        mock_products = mock_products[:2000]
        print(f"Loaded {len(mock_products)} products from {backup_path}.")
    else:
        mock_products = [{
            "_id": f"prod_{i}", "name": f"Libas Women Red Kurta {i}", "category": "kurta",
            "price": 399, "mrp": 899, "discount_percent": 55, "rating": 4.2,
            "review_count": 120, "in_stock": True, "color": ["red"], "tags": ["kurta"],
            "embedding_text": f"Libas Women Red Kurta {i} kurta red"
        } for i in range(100)]
        print(f"Generated {len(mock_products)} inline fallback products.")
else:
    try:
        mongo_client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=2000)
        db = mongo_client[DB_NAME]
        # Ping check
        mongo_client.admin.command('ping')
        print("Connected to MongoDB successfully.")
    except Exception as e:
        print(f"MongoDB connection failed: {e}. Automatically falling back to MOCK_SERVICES mode.")
        MOCK_SERVICES = True
        backup_path = None
        for p in ["../data/products_2k.json", "data/products_2k.json", "../data/products_100k_backup.json", "data/products_100k_backup.json"]:
            if os.path.exists(p):
                backup_path = p
                break
            
        if backup_path:
            with open(backup_path, "r") as f:
                mock_products = json.load(f)
            mock_products = mock_products[:2000]
            print(f"Loaded {len(mock_products)} products from {backup_path}.")
        else:
            mock_products = [{
                "_id": f"prod_{i}", "name": f"Libas Women Red Kurta {i}", "category": "kurta",
                "price": 399, "mrp": 899, "discount_percent": 55, "rating": 4.2,
                "review_count": 120, "in_stock": True, "color": ["red"], "tags": ["kurta"],
                "embedding_text": f"Libas Women Red Kurta {i} kurta red"
            } for i in range(100)]
            print(f"Generated {len(mock_products)} inline fallback products.")

bm25 = BM25Retriever()
faiss = FAISSRetriever()
clip = CLIPRetriever()
ltr_model = LTRModel()

# Flag to trigger incremental index rebuilds
bm25_dirty = False

def rebuild_indices_from_db():
    global bm25_dirty
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
        
        bm25_dirty = False
        print("Indices successfully rebuilt and saved.")
        return True
    except Exception as e:
        print(f"Error building indices: {e}")
        return False

# Setup background scheduler for BM25 (rebuild every 5 mins if dirty)
scheduler = BackgroundScheduler()

def check_and_rebuild_indices():
    global bm25_dirty
    if bm25_dirty:
        print("Dirty flag detected. Triggering scheduled index rebuild...")
        rebuild_indices_from_db()

scheduler.add_job(check_and_rebuild_indices, 'interval', minutes=5)

@app.on_event("startup")
async def startup_event():
    # Start scheduler
    scheduler.start()
    
    # Try loading existing indices
    bm25_loaded = bm25.load()
    faiss_loaded = faiss.load()
    clip_loaded = clip.load()
    
    if not clip_loaded:
        print("CLIP index not found. Visual search will be disabled.")
    
    if not bm25_loaded or not faiss_loaded:
        print("Indices not found or incomplete. Rebuilding from DB...")
        rebuild_indices_from_db()
        # Reload models
        ltr_model.load()

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
            return {
                "query": q,
                "intent": intent.model_dump(),
                "total": 0,
                "results": []
            }

        # Step 4: fetch product details for top 100 merged candidates
        product_ids = [pid for pid, _ in merged]
        products = []
        if MOCK_SERVICES:
            products = [p for p in mock_products if p["_id"] in product_ids]
        else:
            try:
                cursor = db["products"].find({"_id": {"$in": product_ids}})
                products = list(cursor)
                if not products:
                    # A fresh Docker MongoDB volume has no seeded products. The
                    # retrieval indexes still refer to the bundled catalog, so
                    # use it to hydrate candidates instead of returning zero hits.
                    products = [p for p in load_local_catalog() if p["_id"] in product_ids]
            except Exception as e:
                print(f"MongoDB query failed: {e}. Falling back to in-memory mock products.")
                MOCK_SERVICES = True
                products = [p for p in load_local_catalog() if p["_id"] in product_ids]
                
        product_map = {p["_id"]: p for p in products}

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
                fv = build_feature_vector(overridden_intent, product, score, clip_score=product_clip_score)
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

        # Step 6: sort
        ranked = sorted(zip(candidates_to_rank, ltr_scores), key=lambda x: x[1], reverse=True)

        # Step 7: Apply post-filters
        filtered_results = []
        
        # Active filters (UI overrides or LLM extracted intent)
        filter_category = category.lower().strip() if category else (intent.category.lower().strip() if intent.category else None)
        filter_max_price = max_price if max_price is not None else intent.max_price
        filter_min_price = min_price
        filter_min_rating = min_rating if min_rating is not None else intent.min_rating

        for pid, ltr_score in ranked:
            product = product_map[pid]
            
            # Category match check
            if filter_category:
                p_cat = product.get("category", "").lower().strip()
                p_subcat = product.get("subcategory", "").lower().strip()
                p_name = product.get("name", "").lower()
                p_tags = [str(t).lower() for t in product.get("tags", [])]
                
                synonyms = {
                    "kurta": ["kurta", "kurti", "ethnic"],
                    "shirt": ["shirt", "shirts"],
                    "tshirt": ["tshirt", "t-shirt", "tee"],
                    "jeans": ["jeans", "denim"],
                    "dress": ["dress", "frock", "gown"],
                    "shoes": ["shoes", "sneakers", "footwear"]
                }
                
                candidates = [filter_category]
                if filter_category in synonyms:
                    candidates.extend(synonyms[filter_category])
                    
                matched = (p_cat == filter_category) or any(c in p_subcat or c in p_name or c in p_tags for c in candidates)
                if not matched:
                    continue
                
            # Price check
            price = product.get("price", 0)
            if filter_max_price is not None and price > filter_max_price:
                continue
            if filter_min_price is not None and price < filter_min_price:
                continue
                
            # Rating check
            rating = product.get("rating", 0.0)
            if filter_min_rating is not None and rating < filter_min_rating:
                continue
                
            # Stock check
            if in_stock_only and not product.get("in_stock", True):
                continue
                
            # Enrich product dict with score for transparency
            enriched_product = product.copy()
            enriched_product["_score"] = float(ltr_score)
            filtered_results.append(enriched_product)

        # Step 8: paginate
        total_results = len(filtered_results)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        page_results = filtered_results[start_idx:end_idx]

        return {
            "query": q,
            "intent": intent.model_dump(),
            "total": total_results,
            "results": page_results
        }
    except Exception as e:
        print(f"Search endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

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

@app.get("/stats")
def stats():
    return {
        "total_products_indexed": len(bm25.product_ids),
        "faiss_index_size": faiss.index.ntotal,
        "clip_index_size": clip.index.ntotal,
        "clip_model_loaded": clip.model is not None,
        "faiss_exclusion_list_size": len(faiss.exclusion_list),
        "bm25_dirty": bm25_dirty,
        "ltr_model_loaded": ltr_model.model is not None
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
