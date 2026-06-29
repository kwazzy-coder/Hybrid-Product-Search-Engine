import os
import sys
import time
import numpy as np
import lightgbm as lgb
from datetime import datetime

# Add search_service to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'search_service')))

from retrieval.bm25_retriever import BM25Retriever
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.hybrid import reciprocal_rank_fusion
from reranker.features import build_feature_vector, price_match_score
from reranker.ltr_model import LTRModel
from query_understanding import parse_query_local_fallback, QueryIntent

# Generate synthetic products
def generate_sample_products(count=1000):
    products = []
    categories = ["kurta", "shirt", "jeans", "dress", "tshirt"]
    colors = ["red", "blue", "black", "white", "pink", "maroon", "navy"]
    brands = ["Libas", "W", "Aurelia", "Biba", "Sangria", "Roadster"]
    
    for i in range(count):
        cat = categories[i % len(categories)]
        col = colors[i % len(colors)]
        brand = brands[i % len(brands)]
        price = 300 + (i * 17) % 1500
        rating = 3.0 + (i % 21) / 10.0
        review_count = i * 3
        
        name = f"{brand} Women {col.capitalize()} {cat.capitalize()}"
        desc = f"Beautiful {col} {cat} made of premium cotton fabric, perfect for casual and festive occasions."
        tags = [cat, col, brand.lower(), "women", "cotton"]
        embedding_text = f"{name} {desc} {cat} {col} {brand} {' '.join(tags)}"
        
        products.append({
            "_id": f"prod_{1000 + i}",
            "name": name,
            "description": desc,
            "category": cat,
            "color": [col],
            "brand": brand,
            "price": price,
            "mrp": price + 200,
            "discount_percent": 15,
            "rating": rating,
            "review_count": review_count,
            "in_stock": i % 20 != 0, # 5% out of stock
            "tags": tags,
            "embedding_text": embedding_text
        })
    return products

def run_verification():
    print("=== HYBRID SEARCH ENGINE OFFLINE VERIFICATION ===")
    
    # 1. Generate Sample Data
    products = generate_sample_products(1000)
    print(f"Generated {len(products)} synthetic products for testing.")
    
    # 2. Test Query Understanding Fallback
    print("\n--- 1. Testing Query Parsing (Local Fallback) ---")
    queries_to_test = [
        "red kurta under 500",
        "blue shirt above 4 star",
        "black jeans under 1200 in stock"
    ]
    for q in queries_to_test:
        intent = parse_query_local_fallback(q)
        print(f"Query: '{q}'")
        print(f"  Parsed Category: {intent.category}")
        print(f"  Parsed Colors: {intent.color}")
        print(f"  Max Price Limit: {intent.max_price}")
        print(f"  Min Rating Limit: {intent.min_rating}")
        print(f"  Expansion Synonyms: {intent.expanded_terms}")
        
    # 3. Test BM25 Retriever
    print("\n--- 2. Testing BM25 Lexical Indexing & Search ---")
    bm25 = BM25Retriever()
    t0 = time.perf_counter()
    bm25.build(products)
    t1 = time.perf_counter()
    print(f"  Built BM25 index in {(t1 - t0)*1000:.2f} ms")
    
    bm25.save("models/test_bm25.pkl")
    bm25.load("models/test_bm25.pkl")
    
    t0 = time.perf_counter()
    bm25_res = bm25.search(["kurta", "red", "ethnic"], top_k=20)
    t1 = time.perf_counter()
    print(f"  BM25 search completed in {(t1 - t0)*1000:.2f} ms")
    print(f"  BM25 top 3 results: {bm25_res[:3]}")
    
    # 4. Test FAISS Retriever
    print("\n--- 3. Testing FAISS Semantic Indexing & Search ---")
    faiss_ret = FAISSRetriever()
    t0 = time.perf_counter()
    faiss_ret.add_products(products[:100]) # use a smaller set for fast CPU embedding model load
    t1 = time.perf_counter()
    print(f"  Built FAISS index for 100 products in {(t1 - t0)*1000:.2f} ms")
    
    faiss_ret.save("models/test_faiss.index", "models/test_faiss_map.pkl")
    faiss_ret.load("models/test_faiss.index", "models/test_faiss_map.pkl")
    
    t0 = time.perf_counter()
    faiss_res = faiss_ret.search("red ethnic kurta", top_k=20)
    t1 = time.perf_counter()
    print(f"  FAISS search completed in {(t1 - t0)*1000:.2f} ms")
    print(f"  FAISS top 3 results: {faiss_res[:3]}")
    
    # 5. Test RRF Fusion
    print("\n--- 4. Testing Reciprocal Rank Fusion (RRF) ---")
    merged = reciprocal_rank_fusion(bm25_res, faiss_res, k=60)
    print(f"  RRF combined top 5 products: {merged[:5]}")
    
    # 6. Test Feature Builder
    print("\n--- 5. Testing Feature Vector Builder ---")
    intent = parse_query_local_fallback("red kurta under 500")
    prod = products[0]
    fv = build_feature_vector(intent, prod, 0.05)
    print(f"  Features for product '{prod['name']}':")
    print(f"    RRF score: {fv[0]}")
    print(f"    Category match: {fv[1]}")
    print(f"    Color match: {fv[2]}")
    print(f"    Price match score: {fv[3]}")
    print(f"    Normalized rating: {fv[4]}")
    print(f"    Review count (capped): {fv[5]}")
    print(f"    Discount signal: {fv[6]}")
    print(f"    In stock: {fv[7]}")
    
    # 7. Test LightGBM LTR Training offline
    print("\n--- 6. Testing LTR Offline Training ---")
    # Generate mock training vectors
    X_train = []
    y_train = []
    groups_train = []
    
    # Simulate 50 queries, 20 results each = 1000 training interactions
    for q_idx in range(50):
        groups_train.append(20)
        q_intent = parse_query_local_fallback("red kurta")
        for p_idx in range(20):
            p = products[p_idx]
            label = 2 if (p_idx < 3 and p["in_stock"]) else (1 if p_idx < 10 else 0)
            X_train.append(build_feature_vector(q_intent, p, 1.0 / (60 + p_idx)))
            y_train.append(label)
            
    X_np = np.array(X_train, dtype=np.float32)
    y_np = np.array(y_train, dtype=np.int32)
    g_np = np.array(groups_train, dtype=np.int32)
    
    train_data = lgb.Dataset(X_np, label=y_np, group=g_np)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "learning_rate": 0.05,
        "num_leaves": 7,
        "min_data_in_leaf": 2,
        "verbose": -1
    }
    
    t0 = time.perf_counter()
    model = lgb.train(params, train_data, num_boost_round=20)
    t1 = time.perf_counter()
    print(f"  Trained LightGBM LambdaRank model in {(t1 - t0)*1000:.2f} ms")
    
    os.makedirs("models", exist_ok=True)
    model.save_model("models/test_ltr_model.lgb")
    
    # Load and Predict LTR
    ltr = LTRModel(model_path="models/test_ltr_model.lgb")
    p_scores = ltr.predict(X_train[:10])
    print(f"  LTR prediction scores for first 5 items: {p_scores[:5]}")
    
    # 8. Performance Benchmark
    print("\n--- 7. Performance & Latency Benchmarks ---")
    latency_records = []
    
    for _ in range(50):
        t_start = time.perf_counter()
        
        # Simulating endpoint pipeline
        intent = parse_query_local_fallback("red kurta under 500")
        bm25_res = bm25.search(intent.expanded_terms, top_k=100)
        faiss_res = faiss_ret.search(intent.original_query, top_k=100)
        merged = reciprocal_rank_fusion(bm25_res, faiss_res)
        
        product_map = {p["_id"]: p for p in products[:100]}
        feature_vectors = [
            build_feature_vector(intent, product_map[pid], score)
            for pid, score in merged if pid in product_map
        ]
        ltr_scores = ltr.predict(feature_vectors)
        
        # Sort and post-filter simulation
        ranked = sorted(zip([p for p, _ in merged if p in product_map], ltr_scores), key=lambda x: x[1], reverse=True)
        ranked = [p for p, s in ranked if product_map[p]["price"] <= (intent.max_price or 9999)]
        page_results = ranked[:20]
        
        t_end = time.perf_counter()
        latency_records.append((t_end - t_start) * 1000) # in ms
        
    p50 = np.percentile(latency_records, 50)
    p95 = np.percentile(latency_records, 95)
    p99 = np.percentile(latency_records, 99)
    
    print(f"  Pipeline Latency Benchmark (100 candidates):")
    print(f"    P50 Latency: {p50:.2f} ms")
    print(f"    P95 Latency: {p95:.2f} ms")
    print(f"    P99 Latency: {p99:.2f} ms")
    
    if p99 < 50.0:
        print("  SUCCESS: P99 Latency is under 50ms!")
    else:
        print("  WARNING: P99 Latency exceeds 50ms.")

if __name__ == "__main__":
    run_verification()
