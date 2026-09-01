"""
Retrain LTR Model with Synthetic Data & Real Click Feedback
==========================================================
Ingests click feedback (hard negatives from SQLite/data/hard_negatives.json)
alongside synthetic data to train the 12-feature LightGBM LambdaRank model.

Usage:
    python scripts/retrain_ltr.py [--mode hybrid|feedback|synthetic]
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_understanding import QueryIntent, parse_query_local_fallback
from reranker.features import build_feature_vector
from feedback_db import get_all_click_events, init_db

# Data & Model Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SEARCH_SERVICE_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_DIR = os.path.dirname(SEARCH_SERVICE_DIR)

DATA_PATH = os.path.join(PROJECT_DIR, "data", "products_2k.json")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(SEARCH_SERVICE_DIR, "data", "products_2k.json")

HARD_NEGATIVES_PATH = os.path.join(PROJECT_DIR, "data", "hard_negatives.json")
if not os.path.exists(HARD_NEGATIVES_PATH):
    HARD_NEGATIVES_PATH = os.path.join(SEARCH_SERVICE_DIR, "data", "hard_negatives.json")

MODEL_PATH = os.path.join(SEARCH_SERVICE_DIR, "models", "ltr_model.lgb")

SYNTHETIC_QUERIES = [
    "red dress", "blue shirt", "black shoes", "white tshirt", "green kurta",
    "formal shoes", "casual jeans", "sports shoes", "pink dress under 1000",
    "navy blue shirt", "running shoes", "denim jeans", "cotton shirt",
    "summer dress", "winter jacket", "formal shirt", "party dress",
    "leather shoes", "sneakers", "sandals", "ethnic kurta", "casual wear",
    "office wear", "blue jeans", "black formal shoes", "red top",
    "floral dress", "striped shirt", "polo tshirt", "track pants",
]

def load_catalog():
    """Load product catalog."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_synthetic_samples(products: list):
    """Generate heuristic synthetic samples for query bank."""
    all_features = []
    all_labels = []
    all_groups = []

    random.seed(42)
    for query in SYNTHETIC_QUERIES:
        intent = parse_query_local_fallback(query)
        sample_size = min(50, len(products))
        sampled = random.sample(products, sample_size)

        group_features = []
        group_labels = []

        for product in sampled:
            rrf_score = random.uniform(0.005, 0.05)
            clip_score = random.uniform(0.1, 0.5)
            price_comp = random.uniform(0.3, 1.0)

            fv = build_feature_vector(intent, product, rrf_score, clip_score=clip_score, price_competitiveness=price_comp)

            relevance = 0
            cat_match = fv[1]
            color_match = fv[2]
            clip_s = fv[8]
            pc = fv[11]

            if cat_match > 0 and color_match > 0:
                relevance = 3
            elif cat_match > 0:
                relevance = 2
            elif color_match > 0:
                relevance = 1

            if clip_s > 0.35:
                relevance = min(relevance + 1, 3)

            if pc > 0.9 and relevance >= 2:
                relevance = min(relevance + 1, 3)

            group_features.append(fv)
            group_labels.append(relevance)

        all_features.extend(group_features)
        all_labels.extend(group_labels)
        all_groups.append(len(group_features))

    return all_features, all_labels, all_groups

def load_feedback_samples(products: list):
    """Load real click feedback samples from hard_negatives.json or SQLite DB."""
    product_map = {p["_id"]: p for p in products}
    click_events = []

    # Priority 1: Check hard_negatives.json
    if os.path.exists(HARD_NEGATIVES_PATH):
        with open(HARD_NEGATIVES_PATH, "r", encoding="utf-8") as f:
            click_events = json.load(f)
        print(f"Loaded {len(click_events)} feedback sessions from {HARD_NEGATIVES_PATH}")
    else:
        # Priority 2: Direct query SQLite
        init_db()
        events = get_all_click_events()
        if events:
            for idx, e in enumerate(events):
                shown = e.get("shown_results", [])[:10]
                clicked = e.get("clicked_id")
                if not e.get("query") or not clicked:
                    continue
                items = [{"product_id": clicked, "label": 3}]
                for rank, pid in enumerate(shown):
                    if pid != clicked:
                        items.append({"product_id": pid, "label": 0})
                click_events.append({"query": e["query"], "items": items})
            print(f"Loaded {len(click_events)} feedback sessions directly from SQLite database.")

    if not click_events:
        return [], [], []

    all_features = []
    all_labels = []
    all_groups = []

    random.seed(42)
    for session in click_events:
        query = session.get("query")
        items = session.get("items", [])
        if not query or not items:
            continue

        intent = parse_query_local_fallback(query)
        group_features = []
        group_labels = []

        for item in items:
            pid = item.get("product_id")
            label = item.get("label", 0)
            product = product_map.get(pid)
            if not product:
                continue

            rrf_score = random.uniform(0.01, 0.05) if label > 0 else random.uniform(0.001, 0.02)
            clip_score = random.uniform(0.3, 0.7) if label > 0 else random.uniform(0.05, 0.3)
            price_comp = random.uniform(0.5, 1.0)

            fv = build_feature_vector(intent, product, rrf_score, clip_score=clip_score, price_competitiveness=price_comp)

            group_features.append(fv)
            group_labels.append(label)

        if group_features:
            all_features.extend(group_features)
            all_labels.extend(group_labels)
            all_groups.append(len(group_features))

    return all_features, all_labels, all_groups

def main(mode: str = "hybrid"):
    print("=" * 60)
    print(f"  LightGBM LTR Training Pipeline (Mode: {mode.upper()})")
    print("=" * 60)

    products = load_catalog()
    print(f"Loaded {len(products)} catalog items.")

    all_features = []
    all_labels = []
    all_groups = []

    # 1. Ingest Synthetic Heuristic Data
    if mode in ["synthetic", "hybrid"]:
        s_feat, s_lbl, s_grp = generate_synthetic_samples(products)
        all_features.extend(s_feat)
        all_labels.extend(s_lbl)
        all_groups.extend(s_grp)
        print(f"Ingested Synthetic Data: {len(s_feat)} samples across {len(s_grp)} query groups")

    # 2. Ingest Real Click Feedback Data
    if mode in ["feedback", "hybrid"]:
        f_feat, f_lbl, f_grp = load_feedback_samples(products)
        if f_feat:
            all_features.extend(f_feat)
            all_labels.extend(f_lbl)
            all_groups.extend(f_grp)
            print(f"Ingested Real Click Feedback Data: {len(f_feat)} samples across {len(f_grp)} sessions")
        else:
            print("[INFO] No real click feedback data available yet. Using synthetic data.")

    if not all_features:
        print("[ERROR] No training data available!")
        return

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32)

    print(f"\nTotal Training Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(all_groups)} groups")
    unique_lbls, counts = np.unique(y, return_counts=True)
    print(f"Label Distribution: {dict(zip(unique_lbls.tolist(), counts.tolist()))}")

    # Train LightGBM LambdaRank
    train_data = lgb.Dataset(X, label=y, group=all_groups)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "verbose": -1,
    }

    print("\nTraining LightGBM LambdaRank Model ...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data],
        callbacks=[lgb.log_evaluation(50)],
    )

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)
    
    # Save a copy as ltr_model_v2.lgb for version tracking
    versioned_path = os.path.join(os.path.dirname(MODEL_PATH), "ltr_model_v2.lgb")
    model.save_model(versioned_path)

    print("\n" + "=" * 60)
    print("  LTR Model Retrained & Successfully Saved!")
    print("=" * 60)
    print(f"  Primary Model Saved To:   {MODEL_PATH}")
    print(f"  Versioned Model Saved To: {versioned_path}")

    # Feature Importances
    feature_names = [
        "rrf", "cat_match", "color_match", "price", "rating", "reviews",
        "discount", "in_stock", "clip_score", "attr_match", "binding", "price_comp"
    ]
    importance = model.feature_importance()
    print("\nFeature Importance Rankings:")
    for name, imp in sorted(zip(feature_names, importance), key=lambda x: -x[1]):
        print(f"  {name:18s} {imp}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain LTR model with click feedback")
    parser.add_argument("--mode", type=str, choices=["hybrid", "feedback", "synthetic"], default="hybrid",
                        help="Data ingestion mode: hybrid (default), feedback, or synthetic")
    args = parser.parse_args()
    
    main(mode=args.mode)
