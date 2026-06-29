def reciprocal_rank_fusion(
    bm25_results: list[tuple[str, float]],
    faiss_results: list[tuple[str, float]],
    k: int = 60
) -> list[tuple[str, float]]:
    """
    RRF score = sum(1 / (k + rank_i)) across all lists.
    k=60 is the standard constant from the original RRF paper.
    """
    scores = {}

    # Rank is 0-indexed in loop, matching (rank + 1) in equation
    for rank, (product_id, _) in enumerate(bm25_results):
        scores[product_id] = scores.get(product_id, 0) + 1 / (k + rank + 1)

    for rank, (product_id, _) in enumerate(faiss_results):
        scores[product_id] = scores.get(product_id, 0) + 1 / (k + rank + 1)

    # sort by descending RRF score
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged[:100]   # top 100 candidates for reranker
