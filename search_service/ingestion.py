"""
Schema Normalization & Cross-Platform Ingestion Layer
=====================================================
Maps heterogeneous product schemas from multiple e-commerce sources
into a single canonical schema before indexing.
"""

import hashlib
import random
from typing import Optional


# Canonical product schema that all sources normalize into
CANONICAL_FIELDS = [
    "_id", "name", "category", "subcategory", "brand", "color", "tags",
    "price", "mrp", "discount_percent", "rating", "review_count",
    "in_stock", "image_url", "embedding_text", "source", "source_url",
    "source_product_id"
]

# Source-specific schema mappings
SOURCE_SCHEMAS = {
    "amazon": {
        "title": "name",
        "product_category": "category",
        "colour": "color",
        "selling_price": "price",
        "list_price": "mrp",
        "avg_rating": "rating",
        "num_ratings": "review_count",
        "availability": "in_stock",
        "image": "image_url",
    },
    "myntra": {
        "productDisplayName": "name",
        "articleType": "category",
        "baseColour": "color",
        "price": "price",
        "mrp": "mrp",
        "rating": "rating",
        "ratingCount": "review_count",
        "inStock": "in_stock",
        "searchImage": "image_url",
    },
    "ajio": {
        "product_name": "name",
        "brick": "category",
        "primary_colour": "color",
        "offer_price": "price",
        "marked_price": "mrp",
        "average_rating": "rating",
        "total_reviews": "review_count",
        "stock_status": "in_stock",
        "img_link": "image_url",
    },
    "flipkart": {
        "product_name": "name",
        "category": "category",
        "color": "color",
        "discounted_price": "price",
        "actual_price": "mrp",
        "rating": "rating",
        "rating_count": "review_count",
        "is_available": "in_stock",
        "image": "image_url",
    },
}

# Category taxonomy normalization across sources
CATEGORY_ALIASES = {
    # Amazon categories
    "men's shirts": "shirt", "women's shirts": "shirt", "button-down shirts": "shirt",
    "men's t-shirts": "tshirt", "women's t-shirts": "tshirt", "graphic tees": "tshirt",
    "men's jeans": "jeans", "women's jeans": "jeans", "skinny jeans": "jeans",
    "women's dresses": "dress", "maxi dresses": "dress", "cocktail dresses": "dress",
    "running shoes": "shoes", "casual shoes": "shoes", "formal shoes": "shoes",
    "women's kurtas": "kurta", "men's kurtas": "kurta", "kurti": "kurta",
    # Myntra/Ajio categories
    "shirts": "shirt", "tshirts": "tshirt", "tops": "tshirt",
    "kurtas": "kurta", "dresses": "dress",
    "casual shoes": "shoes", "sports shoes": "shoes", "heels": "shoes",
    "sandals": "shoes", "flip flops": "shoes", "sneakers": "shoes",
    "trousers": "jeans", "track pants": "jeans", "shorts": "jeans",
}


def normalize_color(color_val) -> list[str]:
    """Normalize color field from various formats to a list of lowercase strings."""
    if color_val is None:
        return []
    if isinstance(color_val, list):
        return [str(c).strip().lower() for c in color_val if c]
    color_str = str(color_val).strip().lower()
    # Handle comma-separated colors: "red, blue"
    if "," in color_str:
        return [c.strip() for c in color_str.split(",") if c.strip()]
    # Handle slash-separated: "red/blue"
    if "/" in color_str:
        return [c.strip() for c in color_str.split("/") if c.strip()]
    return [color_str] if color_str else []


def normalize_price(price_val) -> float:
    """Extract numeric price from various formats."""
    if price_val is None:
        return 0.0
    if isinstance(price_val, (int, float)):
        return float(price_val)
    # Handle strings like "₹1,299", "$29.99", "1299"
    price_str = str(price_val).replace(",", "").replace("₹", "").replace("$", "").strip()
    try:
        return float(price_str)
    except ValueError:
        return 0.0


def normalize_stock(stock_val) -> bool:
    """Normalize stock/availability to boolean."""
    if isinstance(stock_val, bool):
        return stock_val
    if isinstance(stock_val, str):
        return stock_val.strip().lower() in ["true", "in stock", "available", "yes", "1"]
    if isinstance(stock_val, (int, float)):
        return stock_val > 0
    return True


def normalize_category(category_str: str) -> str:
    """Map source-specific categories to canonical categories."""
    if not category_str:
        return ""
    cat_lower = category_str.strip().lower()
    return CATEGORY_ALIASES.get(cat_lower, cat_lower)


def normalize_product(raw_product: dict, source: str) -> dict:
    """
    Normalize a product from any source into the canonical schema.
    
    Args:
        raw_product: Product dict in source-specific schema
        source: Source identifier (e.g., "amazon", "myntra")
    
    Returns:
        Product dict in canonical schema
    """
    schema = SOURCE_SCHEMAS.get(source, {})
    
    # Map source fields to canonical fields
    mapped = {}
    for source_field, canonical_field in schema.items():
        if source_field in raw_product:
            mapped[canonical_field] = raw_product[source_field]
    
    # Also pick up any fields that already match canonical names
    for field in CANONICAL_FIELDS:
        if field not in mapped and field in raw_product:
            mapped[field] = raw_product[field]
    
    # Generate canonical _id
    source_pid = str(raw_product.get("_id", raw_product.get("id", "")))
    canonical_id = f"{source}_{source_pid}"
    
    # Normalize individual fields
    name = str(mapped.get("name", "Unknown Product"))
    category = normalize_category(str(mapped.get("category", "")))
    color = normalize_color(mapped.get("color"))
    price = normalize_price(mapped.get("price"))
    mrp = normalize_price(mapped.get("mrp", price))
    in_stock = normalize_stock(mapped.get("in_stock", True))
    
    # Compute discount
    discount = int(((mrp - price) / mrp) * 100) if mrp > 0 and mrp > price else 0
    
    # Build embedding text
    embedding_text = f"{name} {category} {' '.join(color)} {' '.join(mapped.get('tags', []))}".strip()
    
    return {
        "_id": canonical_id,
        "name": name,
        "category": category,
        "subcategory": mapped.get("subcategory", ""),
        "brand": mapped.get("brand", ""),
        "color": color,
        "tags": mapped.get("tags", []),
        "price": price,
        "mrp": mrp,
        "discount_percent": discount,
        "rating": float(mapped.get("rating", 0) or 0),
        "review_count": int(mapped.get("review_count", 0) or 0),
        "in_stock": in_stock,
        "image_url": mapped.get("image_url", ""),
        "embedding_text": embedding_text,
        "source": source,
        "source_url": mapped.get("source_url", f"https://{source}.com/product/{source_pid}"),
        "source_product_id": source_pid,
    }


def create_cross_platform_dataset(products: list[dict], duplicate_ratio: float = 0.15) -> list[dict]:
    """
    Take a single-source product list and simulate multi-platform listings.
    
    - Assigns each product to a random source
    - Creates ~15% near-duplicates across sources (same product, different name/price)
    - Returns the expanded multi-source product list
    """
    sources = ["amazon", "myntra", "ajio", "flipkart"]
    rng = random.Random(42)
    
    multi_source_products = []
    
    # Step 1: Assign each product to a primary source
    for product in products:
        source = rng.choice(sources)
        p = product.copy()
        p["source"] = source
        p["source_product_id"] = str(p["_id"])
        p["_id"] = f"{source}_{p['_id']}"
        p["source_url"] = f"https://{source}.com/product/{p['source_product_id']}"
        multi_source_products.append(p)
    
    # Step 2: Create cross-source duplicates (same product, different source/price)
    num_duplicates = int(len(products) * duplicate_ratio)
    duplicate_indices = rng.sample(range(len(products)), min(num_duplicates, len(products)))
    
    for idx in duplicate_indices:
        original = products[idx]
        # Pick a different source
        original_source = multi_source_products[idx]["source"]
        other_sources = [s for s in sources if s != original_source]
        dup_source = rng.choice(other_sources)
        
        dup = original.copy()
        dup["source"] = dup_source
        dup["source_product_id"] = str(dup["_id"])
        dup["_id"] = f"{dup_source}_{dup['_id']}"
        dup["source_url"] = f"https://{dup_source}.com/product/{dup['source_product_id']}"
        
        # Vary price by ±5-15% (different platforms have different prices)
        price_factor = rng.uniform(0.85, 1.15)
        dup["price"] = round(dup["price"] * price_factor)
        dup["mrp"] = round(dup.get("mrp", dup["price"]) * rng.uniform(0.95, 1.05))
        if dup["mrp"] > 0 and dup["mrp"] > dup["price"]:
            dup["discount_percent"] = int(((dup["mrp"] - dup["price"]) / dup["mrp"]) * 100)
        
        # Slightly vary the name (e.g., "Blue Shirt" vs "Men's Blue Shirt")
        name_prefixes = ["", "Premium ", "New ", ""]
        dup["name"] = rng.choice(name_prefixes) + dup["name"]
        
        # Keep same image_url (key for CLIP deduplication!)
        multi_source_products.append(dup)
    
    print(f"  Cross-platform dataset: {len(multi_source_products)} products "
          f"({len(products)} originals + {len(multi_source_products) - len(products)} cross-source duplicates)")
    
    return multi_source_products
