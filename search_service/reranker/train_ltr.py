import os
import sys
import json
import random
import numpy as np
import lightgbm as lgb
from pymongo import MongoClient


# Add parent directory (search_service) to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from query_understanding import parse_query_local_fallback
from features import build_feature_vector

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = "ecommerce"
MODEL_PATH = "models/ltr_model.lgb"

def load_data_from_mongo():
    try:
        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        client.admin.command('ping')
    except Exception as e:
        print(f"Could not connect to MongoDB for training: {e}")
        return None, None
        
    click_logs_coll = db["click_logs"]
    products_coll = db["products"]
    
    logs = list(click_logs_coll.find({}))
    if not logs:
        print("No click logs found in MongoDB.")
        return None, None
        
    print(f"Loaded {len(logs)} click logs from MongoDB.")
    
    # Pre-cache products in memory to speed up feature building
    pids = list(set(log["product_id"] for log in logs))
    products = list(products_coll.find({"_id": {"$in": pids}}))
    product_map = {p["_id"]: p for p in products}
    
    return logs, product_map

def load_data_from_backup():
    print("Trying to load from local backup JSON files...")
    logs_path = "models/clean_click_logs.json"
    if not os.path.exists(logs_path):
        logs_path = "data/click_logs_backup.json"
        if not os.path.exists(logs_path):
            logs_path = "../data/click_logs_backup.json"
            
    products_path = "data/products_100k_backup.json"
    if not os.path.exists(products_path):
        products_path = "../data/products_100k_backup.json"
        
    if not os.path.exists(logs_path) or not os.path.exists(products_path):
        print("Local backup JSON files not found. Creating inline synthetic training data...")
        # Generate inline synthetic data if no files exist
        return generate_synthetic_offline_data()
        
    with open(logs_path, "r") as f:
        logs = json.load(f)
    with open(products_path, "r") as f:
        products = json.load(f)
        
    product_map = {p["_id"]: p for p in products}
    return logs, product_map

def generate_synthetic_offline_data():
    """Generates inline data directly for testing train_ltr.py standalone"""
    logs = []
    product_map = {}
    
    categories = ["kurta", "shirt", "jeans"]
    colors = ["red", "blue", "black"]
    
    # Generate 50 products
    for idx in range(100):
        pid = f"prod_{idx}"
        cat = categories[idx % len(categories)]
        col = colors[idx % len(colors)]
        product_map[pid] = {
            "_id": pid,
            "category": cat,
            "color": [col],
            "price": 400 + (idx * 10) % 1000,
            "rating": 3.0 + (idx % 20) / 10.0,
            "review_count": idx * 10,
            "discount_percent": idx % 50,
            "in_stock": idx % 10 != 0
        }
        
    # Generate logs
    for s_idx in range(50):
        session_id = f"sess_{s_idx}"
        q_info = [("red kurta under 500", "kurta", "red", 500), ("blue shirt", "shirt", "blue", None)][s_idx % 2]
        query_text, category, color, max_price = q_info
        
        # Select 20 products
        candidates = list(product_map.keys())[:20]
        for pos, pid in enumerate(candidates):
            p = product_map[pid]
            is_match = p["category"] == category and color in p["color"]
            
            clicked = False
            purchased = False
            if is_match:
                clicked = random.random() < 0.7
                purchased = clicked and random.random() < 0.4
            else:
                clicked = random.random() < 0.1
                
            logs.append({
                "session_id": session_id,
                "query": query_text,
                "product_id": pid,
                "position_shown": pos + 1,
                "clicked": clicked,
                "purchased": purchased
            })
            
    return logs, product_map


def main():
    logs, product_map = load_data_from_mongo()
    if logs is None or product_map is None:
        logs, product_map = load_data_from_backup()
        
    print(f"Processing features for {len(logs)} logs...")
    
    # Sort logs by session_id to ensure query groups are contiguous
    logs.sort(key=lambda x: x["session_id"])
    
    # Process features
    X = []
    y = []
    groups = []
    
    current_session = None
    current_group_size = 0
    
    # Query intent cache to avoid parsing the same string repeatedly
    intent_cache = {}
    
    skipped = 0
    for log in logs:
        session_id = log["session_id"]
        q_text = log["query"]
        pid = log["product_id"]
        
        if pid not in product_map:
            skipped += 1
            continue
            
        # Get product and parse intent
        product = product_map[pid]
        if q_text not in intent_cache:
            intent_cache[q_text] = parse_query_local_fallback(q_text)
        intent = intent_cache[q_text]
        
        # Estimate historical RRF score based on the position shown
        # RRF formula is 1 / (60 + rank), let's use that
        pos = log["position_shown"]
        rrf_score = 1.0 / (60 + pos)
        
        # Build feature vector
        features = build_feature_vector(intent, product, rrf_score)
        
        # Calculate LTR Label: 2 = purchased, 1 = clicked, 0 = ignored
        label = 0
        if log.get("purchased", False):
            label = 2
        elif log.get("clicked", False):
            label = 1
            
        X.append(features)
        y.append(label)
        
        # Group tracking
        if session_id != current_session:
            if current_session is not None:
                groups.append(current_group_size)
            current_session = session_id
            current_group_size = 1
        else:
            current_group_size += 1
            
    # Don't forget the last group
    if current_group_size > 0:
        groups.append(current_group_size)
        
    print(f"Skipped {skipped} click logs due to missing products in map.")
    print(f"Final training dataset size: X={len(X)}, Groups={len(groups)}")
    
    X_np = np.array(X, dtype=np.float32)
    y_np = np.array(y, dtype=np.int32)
    groups_np = np.array(groups, dtype=np.int32)
    
    # Create LightGBM Dataset
    train_data = lgb.Dataset(X_np, label=y_np, group=groups_np)
    
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 5,
        "verbose": -1
    }
    
    print("Training LightGBM LambdaRank model...")
    # Train the model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100
    )
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"LTR Model successfully saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
