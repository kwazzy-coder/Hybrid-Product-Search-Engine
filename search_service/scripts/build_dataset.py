"""
Build Dataset Pipeline
======================
Downloads fashion product images from HuggingFace, generates synthetic
e-commerce metadata, and builds all 3 search indexes (BM25, FAISS text, CLIP image).

Usage:
    cd search_service
    python scripts/build_dataset.py

Requirements: datasets, transformers, torch, sentence-transformers, Pillow, rank-bm25, faiss-cpu
"""

import os
import sys
import json
import random
import hashlib
import numpy as np
from pathlib import Path

# Add parent dir so we can import retrieval modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUM_PRODUCTS = 2000
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "images")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def generate_synthetic_ecommerce_fields(product_name: str, category: str) -> dict:
    """Generate realistic synthetic price, rating, reviews, etc."""
    seed = int(hashlib.md5(product_name.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    # Price ranges by category
    price_ranges = {
        "topwear": (299, 1999),
        "bottomwear": (399, 2499),
        "shoes": (499, 3499),
        "accessories": (149, 999),
        "dress": (499, 2999),
        "innerwear": (199, 999),
        "bags": (299, 2499),
        "watches": (499, 4999),
        "jewellery": (199, 1999),
        "default": (299, 1999),
    }

    cat_lower = category.lower() if category else "default"
    price_range = price_ranges.get(cat_lower, price_ranges["default"])
    price = rng.randint(*price_range)
    
    # MRP is 20-60% higher
    markup = rng.uniform(1.2, 1.6)
    mrp = int(price * markup)
    discount = int(((mrp - price) / mrp) * 100)

    rating = round(rng.uniform(2.5, 5.0), 1)
    review_count = rng.randint(5, 2000)
    in_stock = rng.random() > 0.1  # 90% in stock

    return {
        "price": price,
        "mrp": mrp,
        "discount_percent": discount,
        "rating": rating,
        "review_count": review_count,
        "in_stock": in_stock,
    }


def load_dataset_from_huggingface():
    """Load fashion product images dataset from HuggingFace using streaming (no disk cache)."""
    print("Loading dataset from HuggingFace (streaming mode): ashraq/fashion-product-images-small ...")
    from datasets import load_dataset

    ds = load_dataset("ashraq/fashion-product-images-small", split="train", streaming=True)

    # Filter for items that have images and valid names
    valid_items = []
    for item in ds:
        if item.get("image") is not None and item.get("productDisplayName"):
            valid_items.append(item)
        if len(valid_items) >= NUM_PRODUCTS:
            break
        if len(valid_items) % 500 == 0 and len(valid_items) > 0:
            print(f"  Loaded {len(valid_items)}/{NUM_PRODUCTS} products...")

    print(f"  Selected {len(valid_items)} valid products.")
    return valid_items


def save_images(items):
    """Save product images as JPG files."""
    print(f"Saving {len(items)} product images to {IMAGES_DIR} ...")
    saved_count = 0
    for item in items:
        product_id = str(item["id"])
        img = item["image"]
        img_path = os.path.join(IMAGES_DIR, f"{product_id}.jpg")
        if not os.path.exists(img_path):
            try:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(img_path, "JPEG", quality=85)
                saved_count += 1
            except Exception as e:
                print(f"  Warning: Failed to save image for {product_id}: {e}")
    print(f"  Saved {saved_count} new images ({len(items) - saved_count} already existed).")


def build_product_records(items):
    """Convert HuggingFace items to product records with synthetic e-commerce data."""
    print("Building product records ...")
    products = []
    for item in items:
        product_id = str(item["id"])
        name = item.get("productDisplayName", f"Fashion Product {product_id}")
        
        # Extract category info
        master_cat = item.get("masterCategory", "Fashion")
        sub_cat = item.get("subCategory", "")
        article_type = item.get("articleType", "")
        base_colour = item.get("baseColour", "")
        gender = item.get("gender", "")
        season = item.get("season", "")
        usage = item.get("usage", "")

        # Build tags
        tags = [t.lower() for t in [article_type, sub_cat, master_cat, usage, season] if t]
        
        # Color as list
        color = [base_colour.lower()] if base_colour else []

        # Category mapping to match existing filter panel
        cat_map = {
            "shirts": "shirt", "tshirts": "tshirt", "tops": "tshirt",
            "kurtas": "kurta", "jeans": "jeans", "dresses": "dress",
            "shoes": "shoes", "sandals": "shoes", "flip flops": "shoes",
            "heels": "shoes", "casual shoes": "shoes", "formal shoes": "shoes",
            "sports shoes": "shoes", "shorts": "jeans", "trousers": "jeans",
        }
        category = cat_map.get(article_type.lower(), article_type.lower()) if article_type else sub_cat.lower()

        # Synthetic e-commerce fields
        ecom = generate_synthetic_ecommerce_fields(name, sub_cat)

        # Build embedding text (for MiniLM text search)
        embedding_text = f"{name} {article_type} {base_colour} {gender} {usage} {season}".strip()

        product = {
            "_id": product_id,
            "name": name,
            "category": category,
            "subcategory": sub_cat.lower(),
            "brand": gender.capitalize() if gender else "Unisex",
            "color": color,
            "tags": tags,
            "image_url": f"/static/images/{product_id}.jpg",
            "embedding_text": embedding_text,
            **ecom,
        }
        products.append(product)

    print(f"  Built {len(products)} product records.")
    return products


def build_clip_index(items, products):
    """Build CLIP image embeddings FAISS index."""
    print("\n=== Building CLIP Image Index ===")
    
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        from PIL import Image
    except ImportError as e:
        print(f"  CLIP dependencies missing: {e}. Skipping CLIP index.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    print("  Loading CLIP ViT-B/32 ...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    # Encode images in batches (only original items have images)
    # Duplicates share the same image, so we encode once and map to all IDs
    batch_size = 32
    all_embeddings = []
    all_ids = []
    
    # Map source_product_id -> list of product _ids (for duplicates)
    spid_to_pids = {}
    for p in products:
        spid = p.get("source_product_id", p["_id"])
        if spid not in spid_to_pids:
            spid_to_pids[spid] = []
        spid_to_pids[spid].append(p["_id"])
    
    print(f"  Encoding {len(items)} unique product images (mapping to {len(products)} product IDs) ...")
    for i in range(0, len(items), batch_size):
        batch_items = items[i:i+batch_size]
        
        images = []
        batch_spids = []
        for item in batch_items:
            img = item["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
            batch_spids.append(str(item["id"]))
        
        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            image_features = model.get_image_features(**inputs)
            # L2 normalize for cosine similarity
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embeddings = image_features.cpu().numpy().astype("float32")
        
        # Map each embedding to ALL product IDs that share this image
        for emb_idx, spid in enumerate(batch_spids):
            pids = spid_to_pids.get(spid, [])
            for pid in pids:
                all_embeddings.append(embeddings[emb_idx:emb_idx+1])
                all_ids.append(pid)
        
        if (i + batch_size) % 200 == 0 or i + batch_size >= len(items):
            print(f"    Encoded {min(i+batch_size, len(items))}/{len(items)} images")

    all_embeddings = np.vstack(all_embeddings)
    print(f"  Embedding matrix shape: {all_embeddings.shape}")

    # Build FAISS index
    import faiss
    import pickle

    dim = all_embeddings.shape[1]  # 512
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    
    id_map = {}
    int_ids = []
    for pid in all_ids:
        fid = int(hash(pid) & 0x7FFFFFFFFFFFFFFF)
        int_ids.append(fid)
        id_map[fid] = pid
    
    int_ids_np = np.array(int_ids, dtype=np.int64)
    index.add_with_ids(all_embeddings, int_ids_np)
    
    # Save
    clip_index_path = os.path.join(MODELS_DIR, "clip.index")
    clip_map_path = os.path.join(MODELS_DIR, "clip_id_map.pkl")
    
    faiss.write_index(index, clip_index_path)
    with open(clip_map_path, "wb") as f:
        pickle.dump((id_map, set()), f)
    
    print(f"  CLIP index saved: {clip_index_path} ({index.ntotal} vectors, {dim}-d)")

    # Cleanup GPU memory
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_text_indexes(products):
    """Build BM25 and FAISS text indexes."""
    print("\n=== Building BM25 Index ===")
    from retrieval.bm25_retriever import BM25Retriever
    
    bm25 = BM25Retriever()
    bm25.build(products)
    bm25.save()
    print(f"  BM25 index built with {len(bm25.product_ids)} products.")

    print("\n=== Building FAISS Text Index (MiniLM-L6-v2) ===")
    from retrieval.faiss_retriever import FAISSRetriever
    
    faiss_retriever = FAISSRetriever()
    faiss_retriever.add_products(products)
    faiss_retriever.save()
    print(f"  FAISS text index built with {faiss_retriever.index.ntotal} vectors.")


def save_product_data(products):
    """Save product records as JSON."""
    output_path = os.path.join(DATA_DIR, "products_2k.json")
    with open(output_path, "w") as f:
        json.dump(products, f)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nProduct data saved: {output_path} ({size_mb:.1f} MB)")


def main():
    print("=" * 60)
    print("  Fashion Product Search — Dataset Builder")
    print("=" * 60)

    # Step 1: Load from HuggingFace
    items = load_dataset_from_huggingface()

    # Step 2: Save images
    save_images(items)

    # Step 3: Build product records
    products = build_product_records(items)

    # Step 4: Create cross-platform dataset (assign sources + create duplicates)
    from ingestion import create_cross_platform_dataset
    products = create_cross_platform_dataset(products, duplicate_ratio=0.15)

    # Step 5: Save product data JSON
    save_product_data(products)

    # Step 6: Build CLIP image index (use original items for images, but map to new IDs)
    build_clip_index(items, products)

    # Step 7: Build BM25 + FAISS text indexes
    build_text_indexes(products)

    print("\n" + "=" * 60)
    print("  All indexes built successfully!")
    print("=" * 60)
    sources = set(p.get("source", "unknown") for p in products)
    originals = sum(1 for p in products if not p["_id"].count("_") > 1)
    print(f"\n  Products: {len(products)} ({len(sources)} sources: {', '.join(sorted(sources))})")
    print(f"  Images:   {IMAGES_DIR}")
    print(f"  Models:   {MODELS_DIR}")
    print(f"  Data:     {DATA_DIR}")
    print(f"\n  Next: cd search_service && uvicorn main:app --reload")


if __name__ == "__main__":
    main()
