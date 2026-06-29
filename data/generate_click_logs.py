import os
import json
import random
import uuid
from datetime import datetime, timedelta
from pymongo import MongoClient

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = "ecommerce"

# Sample queries and their parsed intent properties for generation
QUERIES = [
    {"query": "red kurta under 500", "category": "kurta", "color": "red", "max_price": 500},
    {"query": "blue shirt under 1000", "category": "shirt", "color": "blue", "max_price": 1000},
    {"query": "black jeans under 1500", "category": "jeans", "color": "black", "max_price": 1500},
    {"query": "white tshirt under 400", "category": "tshirt", "color": "white", "max_price": 400},
    {"query": "pink dress under 800", "category": "dress", "color": "pink", "max_price": 800},
    {"query": "yellow kurta under 600", "category": "kurta", "color": "yellow", "max_price": 600},
    {"query": "navy shirt", "category": "shirt", "color": "navy", "max_price": None},
    {"query": "grey tshirt under 500", "category": "tshirt", "color": "grey", "max_price": 500},
    {"query": "maroon dress under 1200", "category": "dress", "color": "maroon", "max_price": 1200},
    {"query": "black jeans", "category": "jeans", "color": "black", "max_price": None}
]

def main():
    print(f"Connecting to MongoDB at {MONGO_URL}...")
    try:
        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        # Force a connection check
        client.admin.command('ping')
    except Exception as e:
        print(f"Could not connect to MongoDB: {e}. Generating a fallback JSON click logs file...")
        generate_offline_click_logs()
        return

    products_coll = db["products"]
    click_logs_coll = db["click_logs"]
    
    total_products = products_coll.count_documents({})
    if total_products == 0:
        print("Warning: MongoDB products collection is empty. Please run seed_products.py first or start the ingestion service.")
        print("Generating a fallback local JSON file for now.")
        generate_offline_click_logs()
        return
        
    print(f"Found {total_products} products in database. Generating click logs...")
    
    click_logs = []
    
    # We want to generate ~20,000 click log interactions for LTR training
    num_sessions = 800
    base_time = datetime.utcnow() - timedelta(days=10)
    
    generated_count = 0
    
    for session_idx in range(num_sessions):
        session_id = f"sess_{uuid.uuid4().hex[:8]}"
        query_info = random.choice(QUERIES)
        q = query_info["query"]
        category = query_info["category"]
        color = query_info["color"]
        max_price = query_info["max_price"]
        
        # Pull matching and non-matching products to simulate search retrieval
        # Retrieve some products of target category
        matching_products = list(products_coll.find({"category": category}).limit(15))
        # Retrieve some random products
        random_products = list(products_coll.aggregate([{"$sample": {"size": 15}}]))
        
        candidates = list({p["_id"]: p for p in (matching_products + random_products)}.values())
        random.shuffle(candidates)
        
        # Limit to top 20 candidates per query session
        candidates = candidates[:20]
        
        # Sort them by a simulated baseline score (RRF mockup) + rating
        # This acts as our "position shown" order
        candidates.sort(key=lambda p: (
            (1.0 if p["category"] == category else 0.0) * 0.5 +
            (1.0 if color in p["color"] else 0.0) * 0.3 +
            (p["rating"] / 5.0) * 0.2
        ), reverse=True)
        
        session_time = base_time + timedelta(minutes=session_idx * 15 + random.randint(0, 10))
        
        for pos, p in enumerate(candidates):
            # Calculate utility of product to calculate click/purchase probability
            cat_match = 1.0 if p["category"] == category else 0.0
            col_match = 1.0 if color in p["color"] else 0.0
            price_match = 1.0
            if max_price:
                if p["price"] > max_price:
                    price_match = 0.0
                else:
                    price_match = 1.0 - (p["price"] / max_price) * 0.3
            
            in_stock = 1.0 if p["in_stock"] else 0.0
            rating_norm = p["rating"] / 5.0
            
            # Overall utility of this product to user
            # Category match is critical, followed by color and price match
            utility = (cat_match * 0.4) + (col_match * 0.3) + (price_match * 0.2) + (rating_norm * 0.1)
            utility *= in_stock # Out of stock has 0 utility
            
            # Position bias: lower positions are less likely to be clicked
            position_decay = 1.0 / (pos + 1)**0.5
            
            click_prob = utility * position_decay
            
            clicked = random.random() < click_prob
            purchased = False
            
            if clicked:
                purchase_prob = utility * 0.4 # Clicks translate to purchases based on utility
                purchased = random.random() < purchase_prob
                
            log_entry = {
                "query": q,
                "product_id": p["_id"],
                "position_shown": pos + 1,
                "clicked": clicked,
                "purchased": purchased,
                "session_id": session_id,
                "timestamp": session_time.isoformat() + "Z"
            }
            
            click_logs.append(log_entry)
            generated_count += 1
            
        if len(click_logs) >= 5000:
            click_logs_coll.insert_many(click_logs)
            click_logs = []
            print(f"Inserted {generated_count} click logs so far...")
            
    if click_logs:
        click_logs_coll.insert_many(click_logs)
        print(f"Inserted total of {generated_count} click logs.")
        
    print("Click logs generation completed in MongoDB.")

def generate_offline_click_logs():
    """Fallback offline logs generator in case MongoDB is offline during scripts testing"""
    print("Generating offline fallback click logs to 'data/click_logs_backup.json'...")
    
    backup_path = "data/products_100k_backup.json"
    products = []
    if os.path.exists(backup_path):
        with open(backup_path, "r") as f:
            products = json.load(f)
            
    if not products:
        print("Warning: products backup not found. Cannot generate realistic click logs.")
        products = [{"_id": f"prod_{i}", "category": "kurta", "rating": 4.0} for i in range(100)]
        
    # Group products by category
    by_category = {}
    for p in products:
        cat = str(p.get("category", "clothing")).lower().strip()
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(p)
        
    click_logs = []
    
    for session_idx in range(500):
        session_id = f"sess_{session_idx}"
        query_info = random.choice(QUERIES)
        q = query_info["query"]
        category = query_info["category"]
        
        # Pull matching products
        matching = by_category.get(category, [])
        if len(matching) < 15:
            matching = products
            
        candidates = random.sample(matching, min(len(matching), 15))
        # Add random items
        others = random.sample(products, min(len(products), 5))
        candidates.extend(others)
        random.shuffle(candidates)
        candidates = candidates[:20]
        
        # Sort them by simulated relevance to represent search result ordering
        candidates.sort(key=lambda p: (
            (1.0 if str(p.get("category")).lower().strip() == category else 0.0) * 0.7 +
            (float(p.get("rating", 3.0)) / 5.0) * 0.3
        ), reverse=True)
        
        for pos, p in enumerate(candidates):
            pid = p["_id"]
            cat_match = str(p.get("category")).lower().strip() == category
            
            clicked = False
            purchased = False
            
            if cat_match:
                clicked = random.random() < (0.6 / (pos + 1)**0.4)
                if clicked:
                    purchased = random.random() < 0.2
            else:
                clicked = random.random() < (0.1 / (pos + 1)**0.4)
                
            click_logs.append({
                "query": q,
                "product_id": pid,
                "position_shown": pos + 1,
                "clicked": clicked,
                "purchased": purchased,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
            
    os.makedirs("data", exist_ok=True)
    with open("data/click_logs_backup.json", "w") as f:
        json.dump(click_logs, f)
    print(f"Saved {len(click_logs)} offline click logs to data/click_logs_backup.json.")

if __name__ == "__main__":
    main()
