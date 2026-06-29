import os
import json
import random
import re
import requests
import pandas as pd
from kafka import KafkaProducer

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
TOPIC_NAME = "product-updates"

# Colors for mutation
COLORS = ["red", "blue", "green", "black", "white", "yellow", "pink", "maroon", "navy", "grey", "beige", "purple"]
SIZES = ["XS", "S", "M", "L", "XL", "XXL"]
BRANDS = ["Libas", "W", "Aurelia", "Biba", "Anouk", "Sangria", "Roadster", "HRX", "Zara", "H&M"]
TAGS_POOL = ["ethnic", "casual", "cotton", "festive", "formal", "sports", "running", "slim-fit", "embroidered", "printed"]

def download_dataset():
    csv_path = "data/flipkart_com-ecommerce_sample.csv"
    if os.path.exists(csv_path):
        print(f"Dataset already exists at {csv_path}")
        return csv_path
    
    os.makedirs("data", exist_ok=True)
    url = "https://huggingface.co/datasets/gcp-acp/flipkart-raw-subset/resolve/main/flipkart_com-ecommerce_sample.csv"
    print(f"Downloading dataset from {url}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(csv_path, "wb") as f:
            f.write(response.content)
        print("Download complete.")
        return csv_path
    except Exception as e:
        print(f"Failed to download dataset: {e}. A synthetic dataset will be generated instead.")
        return None

def generate_synthetic_products(count=20000):
    print("Generating synthetic products database...")
    products = []
    categories = {
        "kurta": ["straight_kurta", "anarkali_kurta", "a_line_kurta"],
        "shirt": ["casual_shirt", "formal_shirt", "denim_shirt"],
        "tshirt": ["round_neck", "polo_tshirt", "v_neck"],
        "jeans": ["slim_fit", "skinny_jeans", "straight_fit"],
        "dress": ["maxi_dress", "bodycon_dress", "a_line_dress"]
    }
    
    for i in range(count):
        cat = random.choice(list(categories.keys()))
        subcat = random.choice(categories[cat])
        brand = random.choice(BRANDS)
        color = random.sample(COLORS, random.randint(1, 2))
        price = random.randint(299, 2999)
        mrp = int(price / (1 - random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5])))
        discount_percent = int(((mrp - price) / mrp) * 100) if mrp > price else 0
        rating = round(random.uniform(3.0, 5.0), 1)
        review_count = random.randint(5, 5000)
        
        name = f"{brand} Women {color[0].capitalize()} {subcat.replace('_', ' ').capitalize()}"
        desc = f"Premium quality cotton {cat} for women, suitable for casual and festive occasions."
        
        prod_id = f"prod_{100000 + i}"
        tags = list(set([cat, subcat, brand.lower(), "women", "cotton"] + random.sample(TAGS_POOL, 2)))
        
        products.append({
            "_id": prod_id,
            "name": name,
            "description": desc,
            "category": cat,
            "subcategory": subcat,
            "brand": brand,
            "price": price,
            "mrp": mrp,
            "discount_percent": discount_percent,
            "rating": rating,
            "review_count": review_count,
            "in_stock": random.random() > 0.05,
            "color": color,
            "size": random.sample(SIZES, random.randint(2, 4)),
            "image_url": "https://images.unsplash.com/photo-1583391733956-3750e0ff4e8b?auto=format&fit=crop&w=400&q=80",
            "tags": tags,
            "embedding_text": f"{name} {desc} {cat} {subcat} {' '.join(color)} {brand} {' '.join(tags)}",
            "created_at": "2024-01-15T10:30:00Z",
            "updated_at": "2024-01-15T10:30:00Z"
        })
    return pd.DataFrame(products)

def clean_category_tree(tree_str):
    try:
        categories = json.loads(tree_str)
        if categories and len(categories) > 0:
            parts = [p.strip() for p in categories[0].split(">>")]
            category = parts[0].lower() if len(parts) > 0 else "clothing"
            subcategory = parts[1].lower().replace(" ", "_") if len(parts) > 1 else "clothing_item"
            return category, subcategory
    except Exception:
        pass
    return "clothing", "clothing_item"

def parse_flipkart_csv(file_path):
    print(f"Parsing CSV at {file_path}...")
    try:
        df = pd.read_csv(file_path, encoding='unicode_escape')
    except Exception as e:
        print(f"Unicode escape reading failed: {e}. Trying ISO-8859-1...")
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        
    products = []
    
    for idx, row in df.iterrows():
        try:
            uniq_id = f"prod_{row['uniq_id'][:10]}" if pd.notna(row['uniq_id']) else f"prod_{idx}"
            name = row['product_name'] if pd.notna(row['product_name']) else "Generic Product"
            desc = row['description'] if pd.notna(row['description']) else ""
            brand = row['brand'] if pd.notna(row['brand']) and row['brand'] != "" else "Generic"
            
            # Category parsing
            tree = row['product_category_tree']
            category, subcategory = clean_category_tree(tree)
            
            # Prices
            mrp = float(row['retail_price']) if pd.notna(row['retail_price']) else 1000.0
            price = float(row['discounted_price']) if pd.notna(row['discounted_price']) else mrp
            if price > mrp:
                mrp = price
            discount_percent = int(((mrp - price) / mrp) * 100) if mrp > 0 else 0
            
            # Rating
            rating_str = str(row['product_rating']) if pd.notna(row['product_rating']) else "0"
            rating = 0.0
            rating_match = re.search(r"(\d+(\.\d+)?)", rating_str)
            if rating_match:
                rating = float(rating_match.group(1))
            if rating == 0.0 or rating > 5.0:
                rating = round(random.uniform(3.5, 4.8), 1)
                
            # Random reviews, stock, sizes
            review_count = random.randint(10, 2500)
            in_stock = random.random() > 0.05
            
            # Try to extract color from description or name
            color_matches = [c for c in COLORS if re.search(rf"\b{c}\b", f"{name} {desc}".lower())]
            color = color_matches if color_matches else [random.choice(COLORS)]
            
            # Sizes
            sizes = random.sample(SIZES, random.randint(2, 4))
            
            # Image URL parsing
            images = []
            if pd.notna(row['image']):
                try:
                    images = json.loads(row['image'])
                except Exception:
                    # sometimes not JSON, regex match URL
                    matches = re.findall(r'http[s]?://[^\s",]+', row['image'])
                    if matches:
                        images = matches
            image_url = images[0] if images else "https://images.unsplash.com/photo-1583391733956-3750e0ff4e8b?auto=format&fit=crop&w=400&q=80"
            
            # Tags
            tags = list(set([category, subcategory, brand.lower()] + color + random.sample(TAGS_POOL, 2)))
            
            # Text for embedding representation
            embedding_text = f"{name} {desc} {category} {subcategory} {' '.join(color)} {brand} {' '.join(tags)}"
            # Truncate to reasonable size
            embedding_text = " ".join(embedding_text.split()[:80])
            
            products.append({
                "_id": uniq_id,
                "name": name,
                "description": desc,
                "category": category,
                "subcategory": subcategory,
                "brand": brand,
                "price": price,
                "mrp": mrp,
                "discount_percent": discount_percent,
                "rating": rating,
                "review_count": review_count,
                "in_stock": in_stock,
                "color": color,
                "size": sizes,
                "image_url": image_url,
                "tags": tags,
                "embedding_text": embedding_text,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            })
        except Exception as e:
            continue
            
    if not products:
        print("Parsing resulted in 0 products. Falling back to synthetic.")
        return generate_synthetic_products()
        
    return pd.DataFrame(products)

def mutate_product(prod, index):
    """Mutates a product record to create a new unique one"""
    new_prod = prod.copy()
    new_prod["_id"] = f"{prod['_id']}_{index}"
    
    # Mutate name, price, color, brand
    color = random.sample(COLORS, random.randint(1, 2))
    brand = random.choice(BRANDS)
    
    # Simple regex swap of brand or color if they exist in name
    name = prod["name"]
    for b in BRANDS:
        name = name.replace(b, brand)
    name = f"{brand} {name}" if brand not in name else name
    
    # Price jitter
    price_jitter = random.uniform(0.9, 1.1)
    new_price = round(prod["price"] * price_jitter)
    new_mrp = round(prod["mrp"] * price_jitter)
    if new_price > new_mrp:
        new_mrp = new_price
        
    new_prod["name"] = name
    new_prod["color"] = color
    new_prod["price"] = new_price
    new_prod["mrp"] = new_mrp
    new_prod["brand"] = brand
    new_prod["rating"] = round(min(5.0, max(1.0, prod["rating"] + random.uniform(-0.3, 0.3))), 1)
    new_prod["review_count"] = int(max(0, prod["review_count"] + random.randint(-100, 100)))
    new_prod["in_stock"] = random.random() > 0.05
    
    tags = list(set([new_prod["category"], new_prod["subcategory"], brand.lower()] + color + random.sample(TAGS_POOL, 2)))
    new_prod["tags"] = tags
    new_prod["embedding_text"] = f"{name} {prod['description']} {new_prod['category']} {new_prod['subcategory']} {' '.join(color)} {brand} {' '.join(tags)}"
    new_prod["embedding_text"] = " ".join(new_prod["embedding_text"].split()[:80])
    
    return new_prod

def scale_to_target(df, target=100000):
    current_count = len(df)
    print(f"Original dataset count: {current_count}. Expanding to {target} products...")
    
    records = df.to_dict(orient="records")
    expanded_records = list(records)
    
    duplicates_needed = target - current_count
    for i in range(duplicates_needed):
        base_prod = random.choice(records)
        mutated = mutate_product(base_prod, i)
        expanded_records.append(mutated)
        if (i + 1) % 20000 == 0:
            print(f"Generated {current_count + i + 1} / {target} products...")
            
    return expanded_records

def main():
    csv_file = download_dataset()
    if csv_file:
        df = parse_flipkart_csv(csv_file)
    else:
        df = generate_synthetic_products()
        
    expanded_products = scale_to_target(df, 100000)
    
    print(f"Connecting to Kafka at {KAFKA_BOOTSTRAP_SERVERS}...")
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    except Exception as e:
        print(f"Could not connect to Kafka: {e}. Writing to a local file instead for backup seed.")
        with open("data/products_100k_backup.json", "w") as f:
            json.dump(expanded_products, f)
        print("Saved seed data locally to data/products_100k_backup.json.")
        return

    print("Seeding products to Kafka topic 'product-updates'...")
    sent = 0
    for idx, p in enumerate(expanded_products):
        # Embed message action
        msg = p.copy()
        msg["_action"] = "upsert"
        
        producer.send(TOPIC_NAME, value=msg)
        sent += 1
        
        if sent % 10000 == 0:
            print(f"Sent {sent} / {len(expanded_products)} products to Kafka...")
            producer.flush()
            
    producer.flush()
    print("Kafka seed complete.")

if __name__ == "__main__":
    main()
