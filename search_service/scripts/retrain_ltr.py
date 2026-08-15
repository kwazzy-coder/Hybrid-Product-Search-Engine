"""
Retrain LTR model with 11 features (was 8).
Generates synthetic training data from the current product set.
"""
import os
import sys
import json
import random
import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_understanding import QueryIntent, parse_query_local_fallback
from reranker.features import build_feature_vector

# Load products
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "products_2k.json")
with open(DATA_PATH, "r") as f:
    products = json.load(f)
print(f"Loaded {len(products)} products.")

# Sample queries
QUERIES = [
    "red dress", "blue shirt", "black shoes", "white tshirt", "green kurta",
    "formal shoes", "casual jeans", "sports shoes", "pink dress under 1000",
    "navy blue shirt", "running shoes", "denim jeans", "cotton shirt",
    "summer dress", "winter jacket", "formal shirt", "party dress",
    "leather shoes", "sneakers", "sandals", "ethnic kurta", "casual wear",
    "office wear", "blue jeans", "black formal shoes", "red top",
    "floral dress", "striped shirt", "polo tshirt", "track pants",
]

random.seed(42)
all_features = []
all_labels = []
all_groups = []

print("Generating training data ...")
for query in QUERIES:
    intent = parse_query_local_fallback(query)
    
    # Sample products for this query group
    sample_size = min(50, len(products))
    sampled = random.sample(products, sample_size)
    
    group_features = []
    group_labels = []
    
    for product in sampled:
        # Simulate RRF score and CLIP score
        rrf_score = random.uniform(0.005, 0.05)
        clip_score = random.uniform(0.1, 0.5)
        
        fv = build_feature_vector(intent, product, rrf_score, clip_score=clip_score)
        
        # Generate relevance label based on feature signals
        # Strong positive signals: category match + color match + high clip score
        relevance = 0
        cat_match = fv[1]  # category_match
        color_match = fv[2]  # color_match
        clip_s = fv[8]  # clip_score
        attr_match = fv[9]  # attribute_match
        
        if cat_match > 0 and color_match > 0:
            relevance = 3  # Highly relevant
        elif cat_match > 0:
            relevance = 2  # Relevant
        elif color_match > 0:
            relevance = 1  # Partially relevant
        
        # Boost by CLIP score
        if clip_s > 0.35:
            relevance = min(relevance + 1, 3)
        
        group_features.append(fv)
        group_labels.append(relevance)
    
    all_features.extend(group_features)
    all_labels.extend(group_labels)
    all_groups.append(len(group_features))

X = np.array(all_features, dtype=np.float32)
y = np.array(all_labels, dtype=np.float32)

print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features, {len(all_groups)} query groups")
print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

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

print("\nTraining LightGBM LambdaRank model ...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=200,
    valid_sets=[train_data],
    callbacks=[lgb.log_evaluation(50)],
)

# Save model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "ltr_model.lgb")
model.save_model(MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")

# Print feature importance
feature_names = ["rrf", "cat_match", "color_match", "price", "rating", "reviews", "discount", "in_stock", "clip_score", "attr_match", "binding"]
importance = model.feature_importance()
print("\nFeature importance:")
for name, imp in sorted(zip(feature_names, importance), key=lambda x: -x[1]):
    print(f"  {name:15s} {imp}")
