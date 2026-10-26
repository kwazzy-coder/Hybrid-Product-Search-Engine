"""
The ranking pipeline, shared by the API (main.py), the offline evaluation
(scripts/eval_live.py) and LTR retraining (scripts/train_ltr_live.py).

Keeping one implementation means the numbers you evaluate and the features you
train on are exactly what users get (no train/serve skew).

    retrieve (BM25, MiniLM, CLIP)  ->  fuse (weighted RRF)  ->  build features
    ->  LTR score  ->  source diversity  ->  filters
"""

from __future__ import annotations

from dataclasses import dataclass

from deduplication import diversify_results, price_competitiveness_score
from query_understanding import QueryIntent, category_matches
from reranker.features import build_feature_vector
from retrieval.hybrid import reciprocal_rank_fusion
from sizes import size_matches

TEXT_WEIGHTS = [1.0, 1.0, 0.8]          # BM25, MiniLM, CLIP text->image
IMAGE_TEXT_WEIGHTS = [0.8, 0.8, 1.2]    # photo + text: lean on the image


@dataclass
class Filters:
    category: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    min_rating: float | None = None
    in_stock_only: bool = True
    size: str | None = None


@dataclass
class Candidate:
    product_id: str
    fused_score: float
    features: list[float]
    score: float = 0.0


def resolve_filters(intent: QueryIntent, explicit: Filters, known_categories: set[str]) -> Filters:
    """UI filters win; otherwise use what the query asked for. A parsed category only
    filters if it's one we know (an unknown guess still helps ranking via features)."""
    category = explicit.category.lower().strip() if explicit.category else None
    if not category and intent.category:
        guessed = intent.category.lower().strip()
        category = guessed if guessed in known_categories else None
    return Filters(
        category=category,
        min_price=explicit.min_price,
        max_price=explicit.max_price if explicit.max_price is not None else intent.max_price,
        min_rating=explicit.min_rating if explicit.min_rating is not None else intent.min_rating,
        in_stock_only=explicit.in_stock_only,
        size=explicit.size or getattr(intent, "size", None),
    )


def retrieve(intent: QueryIntent, bm25, faiss, clip) -> tuple[list, list, list]:
    """Sequential retrieval (the API runs the same three calls in parallel threads)."""
    return (
        bm25.search(intent.expanded_terms),
        faiss.search(intent.original_query),
        clip.search(intent.original_query),
    )


def fuse(bm25_results, faiss_results, clip_results, weights=TEXT_WEIGHTS):
    return reciprocal_rank_fusion(bm25_results, faiss_results, clip_results, weights=weights)


def build_candidates(intent, fused, product_map, clip_scores, dedup_index, filters: Filters) -> list[Candidate]:
    feature_intent = intent.model_copy(update={"max_price": filters.max_price, "min_rating": filters.min_rating})
    candidates = []
    for pid, fused_score in fused:
        product = product_map.get(pid)
        if product is None:
            continue
        dedup_info = dedup_index.get(pid)
        pc_score = price_competitiveness_score(product, dedup_info) if dedup_info else 0.5
        fv = build_feature_vector(feature_intent, product, fused_score,
                                  clip_score=clip_scores.get(pid, 0.0), price_competitiveness=pc_score)
        candidates.append(Candidate(pid, fused_score, fv))
    return candidates


def score_candidates(candidates: list[Candidate], ltr_model) -> list[Candidate]:
    if not candidates:
        return []
    try:
        scores = ltr_model.predict([c.features for c in candidates]) if ltr_model is not None else None
    except Exception as exc:
        print(f"LTR prediction error: {exc}")
        scores = None
    for c, s in zip(candidates, scores or [c.fused_score for c in candidates]):
        c.score = float(s)
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def diversify(candidates: list[Candidate], product_map, lambda_diversity: float = 0.3) -> list[Candidate]:
    by_id = {c.product_id: c for c in candidates}
    order = diversify_results([(c.product_id, c.score) for c in candidates], product_map, lambda_diversity=lambda_diversity)
    return [by_id[pid] for pid, _ in order]


def passes_filters(product: dict, filters: Filters) -> bool:
    if filters.category and not category_matches(product, filters.category):
        return False
    price = product.get("price") or 0
    if filters.max_price is not None and price > filters.max_price:
        return False
    if filters.min_price is not None and price < filters.min_price:
        return False
    if filters.min_rating is not None and (product.get("rating") or 0.0) < filters.min_rating:
        return False
    if filters.in_stock_only and not product.get("in_stock", True):
        return False
    if filters.size and not size_matches(product, filters.size):
        return False
    return True


def rank(intent, fused, product_map, clip_scores, ltr_model, dedup_index, filters: Filters,
         use_ltr: bool = True, use_diversity: bool = True, use_filters: bool = True) -> list[Candidate]:
    """Full ranking as served. The switches let the evaluation measure each stage separately."""
    candidates = build_candidates(intent, fused, product_map, clip_scores, dedup_index, filters)
    candidates = score_candidates(candidates, ltr_model if use_ltr else None)
    if use_diversity:
        candidates = diversify(candidates, product_map)
    if use_filters:
        candidates = [c for c in candidates if passes_filters(product_map[c.product_id], filters)]
    return candidates


def rank_photo(intent, visual, product_map, dedup_index, filters: Filters, has_text: bool,
               text_boost: float = 0.04, margin: float = 0.20, min_results: int = 8) -> list[Candidate]:
    """
    Photo search: order by visual similarity (CLIP cosine).

    The LTR model is trained on text-search signals and ignores how similar two images
    look, so it isn't used here. Text typed alongside the photo narrows the results
    (price, size, category filters) and nudges matching category/colour up slightly.
    Results far below the best match are dropped so unrelated items don't pad the list.
    """
    candidates = build_candidates(intent, visual, product_map, dict(visual), dedup_index, filters)
    for c in candidates:
        bonus = text_boost * (c.features[1] + c.features[2]) if has_text else 0.0   # category, colour match
        c.score = c.fused_score + bonus
    candidates.sort(key=lambda c: c.score, reverse=True)
    candidates = [c for c in candidates if passes_filters(product_map[c.product_id], filters)]
    if candidates:
        best = max(c.fused_score for c in candidates)
        kept = [c for c in candidates if c.fused_score >= best - margin]
        candidates = kept if len(kept) >= min_results else candidates[:min_results]
    return candidates
