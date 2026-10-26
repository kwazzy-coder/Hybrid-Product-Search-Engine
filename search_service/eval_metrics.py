"""Ranking metrics shared by the offline evaluation and LTR retraining."""

import math


def dcg(relevances, k: int) -> float:
    return sum((2.0 ** rel - 1.0) / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_at_k(ranked_ids, judgments: dict, k: int) -> float | None:
    """judgments: {product_id: graded relevance}. None when the query has no relevant items."""
    ideal = dcg(sorted(judgments.values(), reverse=True), k)
    if ideal == 0:
        return None
    return dcg([judgments.get(pid, 0) for pid in ranked_ids], k) / ideal


def recall_at_k(ranked_ids, judgments: dict, k: int, threshold: int = 2) -> float | None:
    relevant = {pid for pid, rel in judgments.items() if rel >= threshold}
    if not relevant:
        return None
    return len(relevant & set(ranked_ids[:k])) / len(relevant)


def reciprocal_rank(ranked_ids, judgments: dict, threshold: int = 2) -> float:
    for i, pid in enumerate(ranked_ids):
        if judgments.get(pid, 0) >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def judged_fraction(ranked_ids, judgments: dict, k: int) -> float:
    """How many of the top-k results have a label. Low values mean the metric is unreliable."""
    top = ranked_ids[:k]
    return sum(pid in judgments for pid in top) / len(top) if top else 1.0


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def percentile(values, pct: float) -> float | None:
    values = sorted(values)
    if not values:
        return None
    idx = min(len(values) - 1, max(0, int(round(pct / 100 * (len(values) - 1)))))
    return values[idx]
