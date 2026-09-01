"""
Offline Evaluation Harness for Hybrid AI Product Search Engine
===============================================================
Evaluates search retrieval & ranking pipeline against a ground-truth test set (data/eval_testset.json).

Metrics:
- NDCG@5
- NDCG@10

Modes Compared:
- Re-ranking OFF (Raw RRF Fusion order)
- Re-ranking ON  (Full pipeline with 12-feature LightGBM LambdaRank + Diversity)

Outputs:
- Printed summary comparison table in console
- Detailed query-level results saved to data/eval_results.json

Usage:
    python scripts/evaluate.py
"""

import os
import sys
import json
import math
import numpy as np

# Add parent directory to sys.path so we can import retrieval & reranker modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SEARCH_SERVICE_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_DIR = os.path.dirname(SEARCH_SERVICE_DIR)

sys.path.insert(0, SEARCH_SERVICE_DIR)

from query_understanding import parse_query_local_fallback
from retrieval.bm25_retriever import BM25Retriever
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.clip_retriever import CLIPRetriever
from retrieval.hybrid import reciprocal_rank_fusion
from reranker.features import build_feature_vector
from reranker.ltr_model import LTRModel
from deduplication import find_cross_source_duplicates, build_dedup_index, price_competitiveness_score, diversify_results
from scripts.build_eval_testset import build_eval_testset

# Paths
DATA_DIR = os.path.join(PROJECT_DIR, "data")
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.join(SEARCH_SERVICE_DIR, "data")

TESTSET_PATH = os.path.join(DATA_DIR, "eval_testset.json")
RESULTS_PATH = os.path.join(DATA_DIR, "eval_results.json")
DATA_PATH = os.path.join(DATA_DIR, "products_2k.json")


def dcg_at_k(rel_scores: list, k: int) -> float:
    """Discounted Cumulative Gain at position K."""
    dcg = 0.0
    for i, rel in enumerate(rel_scores[:k]):
        dcg += (2.0 ** rel - 1.0) / math.log2(i + 2)
    return dcg

def ndcg_at_k(ranked_pids: list, ground_truth: dict, k: int) -> float:
    """Normalized Discounted Cumulative Gain at position K."""
    # Actual DCG@K
    actual_rels = [ground_truth.get(pid, 0) for pid in ranked_pids[:k]]
    actual_dcg = dcg_at_k(actual_rels, k)

    # Ideal DCG@K
    all_rels = sorted(list(ground_truth.values()), reverse=True)
    ideal_dcg = dcg_at_k(all_rels, k)

    if ideal_dcg == 0.0:
        return 1.0 if actual_dcg == 0.0 else 0.0
    return actual_dcg / ideal_dcg

def load_catalog():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        products = json.load(f)
    return {p["_id"]: p for p in products}

def run_evaluation():
    print("=" * 80)
    print("  Offline Evaluation Harness — Hybrid Product Search Engine")
    print("=" * 80)

    # 1. Ensure test set exists
    if not os.path.exists(TESTSET_PATH):
        print("Test set not found. Generating data/eval_testset.json ...")
        build_eval_testset()

    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        testset = json.load(f)
    print(f"Loaded {len(testset)} evaluation test queries.")

    # 2. Load catalog & index components
    product_map = load_catalog()
    print(f"Loaded {len(product_map)} products in catalog.")

    print("\nInitializing retrieval & ranking models ...")
    bm25 = BM25Retriever()
    bm25.load()

    faiss_ret = FAISSRetriever()
    faiss_ret.load()

    clip_ret = CLIPRetriever()
    clip_ret.load()

    ltr_model = LTRModel()
    ltr_model.load()

    # Deduplication & cross-platform index
    dup_groups = find_cross_source_duplicates(clip_ret, list(product_map.values()))
    dedup_index = build_dedup_index(list(product_map.values()), dup_groups)

    query_results = []
    
    rrf_ndcg5_list, rrf_ndcg10_list = [], []
    ltr_ndcg5_list, ltr_ndcg10_list = [], []

    print("\nRunning evaluation on test queries ...\n")

    for item in testset:
        query = item["query"]
        ground_truth = item.get("relevance_judgements", {})

        # Step A: Query intent parsing
        intent = parse_query_local_fallback(query)

        # Step B: 3-Way Parallel Retrieval
        bm25_res = bm25.search(intent.expanded_terms)
        faiss_res = faiss_ret.search(intent.original_query)
        clip_res = clip_ret.search(intent.original_query)

        # Step C: 3-Way RRF Fusion (Mode: Re-ranking OFF)
        merged_rrf = reciprocal_rank_fusion(
            bm25_res, faiss_res, clip_res,
            weights=[1.0, 1.0, 0.8]
        )
        rrf_ranked_pids = [pid for pid, _ in merged_rrf]

        # Compute RRF (OFF) metrics
        rrf_ndcg5 = ndcg_at_k(rrf_ranked_pids, ground_truth, 5)
        rrf_ndcg10 = ndcg_at_k(rrf_ranked_pids, ground_truth, 10)

        # Step D: LightGBM LTR Re-ranking (Mode: Re-ranking ON)
        clip_score_map = {pid: score for pid, score in clip_res}
        feature_vectors = []
        candidates_to_rank = []

        for pid, rrf_score in merged_rrf:
            if pid in product_map:
                product = product_map[pid]
                product_clip_score = clip_score_map.get(pid, 0.0)
                dedup_info = dedup_index.get(pid)
                pc_score = price_competitiveness_score(product, dedup_info) if dedup_info else 0.5

                fv = build_feature_vector(intent, product, rrf_score, clip_score=product_clip_score, price_competitiveness=pc_score)
                feature_vectors.append(fv)
                candidates_to_rank.append(pid)

        if feature_vectors and ltr_model.model:
            ltr_scores = ltr_model.predict(feature_vectors)
        else:
            ltr_scores = [fv[0] for fv in feature_vectors]

        ltr_candidates = sorted(zip(candidates_to_rank, ltr_scores), key=lambda x: x[1], reverse=True)
        ltr_diversified = diversify_results(ltr_candidates, product_map, lambda_diversity=0.3)
        ltr_ranked_pids = [pid for pid, _ in ltr_diversified]

        # Compute LTR (ON) metrics
        ltr_ndcg5 = ndcg_at_k(ltr_ranked_pids, ground_truth, 5)
        ltr_ndcg10 = ndcg_at_k(ltr_ranked_pids, ground_truth, 10)

        rrf_ndcg5_list.append(rrf_ndcg5)
        rrf_ndcg10_list.append(rrf_ndcg10)
        ltr_ndcg5_list.append(ltr_ndcg5)
        ltr_ndcg10_list.append(ltr_ndcg10)

        # Calculate Lift
        lift_ndcg5 = ((ltr_ndcg5 - rrf_ndcg5) / rrf_ndcg5 * 100.0) if rrf_ndcg5 > 0 else 0.0
        lift_ndcg10 = ((ltr_ndcg10 - rrf_ndcg10) / rrf_ndcg10 * 100.0) if rrf_ndcg10 > 0 else 0.0

        query_results.append({
            "query": query,
            "rrf_off": {"ndcg@5": round(rrf_ndcg5, 4), "ndcg@10": round(rrf_ndcg10, 4)},
            "ltr_on": {"ndcg@5": round(ltr_ndcg5, 4), "ndcg@10": round(ltr_ndcg10, 4)},
            "lift_pct": {"ndcg@5": round(lift_ndcg5, 2), "ndcg@10": round(lift_ndcg10, 2)}
        })

    # Summary Statistics
    mean_rrf_ndcg5 = float(np.mean(rrf_ndcg5_list))
    mean_rrf_ndcg10 = float(np.mean(rrf_ndcg10_list))
    mean_ltr_ndcg5 = float(np.mean(ltr_ndcg5_list))
    mean_ltr_ndcg10 = float(np.mean(ltr_ndcg10_list))

    total_lift_ndcg5 = ((mean_ltr_ndcg5 - mean_rrf_ndcg5) / mean_rrf_ndcg5 * 100.0) if mean_rrf_ndcg5 > 0 else 0.0
    total_lift_ndcg10 = ((mean_ltr_ndcg10 - mean_rrf_ndcg10) / mean_rrf_ndcg10 * 100.0) if mean_rrf_ndcg10 > 0 else 0.0

    # Print Console Table
    print("=" * 88)
    print(f"{'QUERY':<24} | {'RRF (OFF) N@5':<13} | {'LTR (ON) N@5':<13} | {'N@5 LIFT':<10} | {'LTR (ON) N@10':<13}")
    print("-" * 88)

    for q_res in query_results:
        q_str = q_res["query"][:23]
        off_5 = q_res["rrf_off"]["ndcg@5"]
        on_5 = q_res["ltr_on"]["ndcg@5"]
        lift_5 = q_res["lift_pct"]["ndcg@5"]
        on_10 = q_res["ltr_on"]["ndcg@10"]
        lift_str = f"+{lift_5:.1f}%" if lift_5 >= 0 else f"{lift_5:.1f}%"

        print(f"{q_str:<24} | {off_5:<13.4f} | {on_5:<13.4f} | {lift_str:<10} | {on_10:<13.4f}")

    print("=" * 88)
    print(f"{'MEAN AGGREGATE METRICS':<24} | {mean_rrf_ndcg5:<13.4f} | {mean_ltr_ndcg5:<13.4f} | {f'+{total_lift_ndcg5:.1f}%':<10} | {mean_ltr_ndcg10:<13.4f}")
    print("=" * 88)

    # Save to JSON report
    report = {
        "summary": {
            "num_queries": len(testset),
            "rrf_off_mean": {"ndcg@5": round(mean_rrf_ndcg5, 4), "ndcg@10": round(mean_rrf_ndcg10, 4)},
            "ltr_on_mean": {"ndcg@5": round(mean_ltr_ndcg5, 4), "ndcg@10": round(mean_ltr_ndcg10, 4)},
            "aggregate_lift_pct": {"ndcg@5": round(total_lift_ndcg5, 2), "ndcg@10": round(total_lift_ndcg10, 2)}
        },
        "query_details": query_results
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nEvaluation complete! Detailed results saved to: {RESULTS_PATH}\n")

if __name__ == "__main__":
    run_evaluation()
