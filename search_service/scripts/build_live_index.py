"""
Build the search indexes for the crawled (live) catalog.

    python -m crawler refresh                  # from the project root: crawl + export + thumbnails
    cd search_service
    python scripts/build_live_index.py         # BM25 + MiniLM text vectors + CLIP image vectors + store matching
    CATALOG=live uvicorn main:app --port 8000  # (CATALOG=live is the default once these indexes exist)

Indexes go to search_service/models_live/, so the original demo indexes in
search_service/models/ stay untouched (switch back with CATALOG=demo).
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

SERVICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SERVICE_DIR)

import numpy as np  # noqa: E402

from paths import LIVE_INDEX_DIR, PROJECT_DIR, find_data_file  # noqa: E402
from retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from retrieval.faiss_retriever import FAISSRetriever  # noqa: E402
from retrieval.clip_retriever import CLIPRetriever  # noqa: E402
from deduplication import match_cross_store  # noqa: E402


def image_file(images_dir: str, product_id: str) -> str:
    import re
    return os.path.join(images_dir, re.sub(r"[^A-Za-z0-9_.-]", "_", product_id) + ".jpg")


def build_clip(products, images_dir, out_dir, batch_size=32):
    """Encode local thumbnails with CLIP. Returns {product_id: vector} (empty if CLIP is unavailable)."""
    try:
        import torch  # noqa: F401
        from PIL import Image
    except ImportError as exc:
        print(f"  CLIP skipped: {exc}. Install torch + transformers to enable visual search.")
        return {}

    clip = CLIPRetriever()
    clip._load_model()
    if clip.model is None:
        print("  CLIP skipped: model could not be loaded.")
        return {}

    have = [p for p in products if os.path.exists(image_file(images_dir, p["_id"]))]
    missing = len(products) - len(have)
    print(f"  {len(have)} products have thumbnails" + (f" ({missing} missing — run `python -m crawler images`)" if missing else ""))
    vectors = {}
    for start in range(0, len(have), batch_size):
        batch = have[start:start + batch_size]
        images, ids = [], []
        for p in batch:
            try:
                images.append(Image.open(image_file(images_dir, p["_id"])).convert("RGB"))
                ids.append(p["_id"])
            except Exception:
                continue
        if not images:
            continue
        emb = clip.encode_images(images)
        clip.add_image_embeddings(emb, ids)
        vectors.update(zip(ids, emb))
        done = min(start + batch_size, len(have))
        if done % 320 == 0 or done == len(have):
            print(f"    encoded {done}/{len(have)} images")
    clip.save(os.path.join(out_dir, "clip.index"), os.path.join(out_dir, "clip_id_map.pkl"))
    return vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default=find_data_file("products_live.json") or os.path.join(PROJECT_DIR, "data", "products_live.json"))
    parser.add_argument("--images-dir", default=os.path.join(PROJECT_DIR, "data", "images_live"))
    parser.add_argument("--out", default=LIVE_INDEX_DIR)
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--image-threshold", type=float, default=0.92, help="CLIP cosine needed to call two listings the same product")
    args = parser.parse_args()

    if not os.path.exists(args.catalog):
        sys.exit(f"Catalog not found: {args.catalog}\nRun `python -m crawler refresh` from the project root first.")
    with open(args.catalog, encoding="utf-8") as fh:
        products = json.load(fh)
    os.makedirs(args.out, exist_ok=True)
    print(f"Building live indexes for {len(products)} products from {args.catalog}")
    print(f"  stores: {dict(Counter(p.get('source_name', p.get('source')) for p in products))}")

    print("\n== BM25")
    bm25 = BM25Retriever()
    bm25.build(products)
    bm25.save(os.path.join(args.out, "bm25.pkl"))

    print("\n== Text vectors (all-MiniLM-L6-v2)")
    text_index = FAISSRetriever()
    text_index.add_products(products)
    if text_index.model is None:
        print("  WARNING: sentence-transformers unavailable, so these are placeholder vectors and semantic search "
              "will be meaningless. pip install sentence-transformers and re-run.")
    text_index.save(os.path.join(args.out, "faiss.index"), os.path.join(args.out, "faiss_id_map.pkl"))

    vectors = {}
    if not args.skip_clip:
        print("\n== CLIP image vectors")
        vectors = build_clip(products, args.images_dir, args.out)

    print("\n== Cross-store matching")
    groups = match_cross_store(products, vectors, image_threshold=args.image_threshold) if vectors else {}
    with open(os.path.join(args.out, "matches.json"), "w", encoding="utf-8") as fh:
        json.dump(groups, fh, indent=1)
    print(f"  {len(groups)} products found in more than one store"
          + ("" if vectors else " (needs CLIP vectors)"))
    for gid, members in list(groups.items())[:5]:
        names = {p["_id"]: (p["source_name"], p["name"], p["price"]) for p in products if p["_id"] in members}
        print(f"    {gid}: " + " | ".join(f"{s}: {n} ₹{pr:g}" for s, n, pr in names.values()))

    metadata = {
        "version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "catalog_mode": "live",
        "catalog_path": os.path.abspath(args.catalog),
        "document_count": len(products),
        "clip_vectors": len(vectors),
        "match_groups": len(groups),
        "sources": dict(Counter(p.get("source") for p in products)),
        "last_scraped_at": max((p.get("scraped_at", "") for p in products), default=""),
        "rebuilt_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(args.out, "index_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\nDone. Indexes in {args.out}")
    print("Start the API with:  uvicorn main:app --port 8000   (uses the live catalog automatically)")


if __name__ == "__main__":
    main()
