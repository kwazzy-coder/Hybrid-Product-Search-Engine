"""
Build Labeled Evaluation Test Set (data/eval_testset.json)
=========================================================
Generates 25 test queries with ground-truth relevance scores (0-3 scale)
assigned to catalog products.

Relevance Scale:
- 3: Highly Relevant (exact category and color match)
- 2: Relevant (category match)
- 1: Marginally Relevant (color/attribute match)
- 0: Irrelevant
"""

import os
import sys
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SEARCH_SERVICE_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_DIR = os.path.dirname(SEARCH_SERVICE_DIR)

sys.path.insert(0, SEARCH_SERVICE_DIR)
from query_understanding import parse_query_local_fallback

DATA_PATH = os.path.join(PROJECT_DIR, "data", "products_2k.json")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(SEARCH_SERVICE_DIR, "data", "products_2k.json")

TEST_QUERIES = [
    "red dress", "blue shirt", "black shoes", "white tshirt", "green kurta",
    "formal shoes", "casual jeans", "sports shoes", "pink dress", "navy blue shirt",
    "running shoes", "denim jeans", "cotton shirt", "summer dress", "winter jacket",
    "formal shirt", "party dress", "leather shoes", "sneakers", "sandals",
    "ethnic kurta", "casual wear", "office wear", "blue jeans", "black formal shoes"
]

def build_eval_testset():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        products = json.load(f)

    testset = []

    for query in TEST_QUERIES:
        intent = parse_query_local_fallback(query)
        q_cat = (intent.category or "").lower()
        q_colors = [c.lower() for c in (intent.color or [])]

        relevance_judgements = {}

        for p in products:
            pid = p["_id"]
            p_name = str(p.get("name", "")).lower()
            p_cat = str(p.get("category", "")).lower()
            p_subcat = str(p.get("subcategory", "")).lower()

            p_color_val = p.get("color", [])
            if not isinstance(p_color_val, list):
                p_color_val = [p_color_val]
            p_colors = [str(c).lower() for c in p_color_val if c]

            cat_match = bool(q_cat and (q_cat == p_cat or q_cat in p_subcat or q_cat in p_name))
            color_match = bool(any(c in p_colors for c in q_colors) or any(c in p_name for c in q_colors))

            if cat_match and color_match:
                rel = 3  # Highly relevant
            elif cat_match:
                rel = 2  # Relevant
            elif color_match:
                rel = 1  # Marginally relevant
            else:
                rel = 0  # Irrelevant

            if rel > 0:
                relevance_judgements[pid] = rel

        testset.append({
            "query": query,
            "relevance_judgements": relevance_judgements
        })

    out_path = os.path.join(PROJECT_DIR, "data", "eval_testset.json")
    if not os.path.exists(os.path.dirname(out_path)):
        out_path = os.path.join(SEARCH_SERVICE_DIR, "data", "eval_testset.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(testset, f, indent=2)

    print(f"Evaluation test set built with {len(testset)} queries -> {out_path}")
    return out_path

if __name__ == "__main__":
    build_eval_testset()
