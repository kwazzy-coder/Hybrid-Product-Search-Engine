def reciprocal_rank_fusion(
    *result_lists: list[tuple[str, float]],
    k: int = 60,
    weights: list[float] | None = None
) -> list[tuple[str, float]]:
    """
    N-way weighted RRF. 
    weights defaults to [1.0, 1.0, ...] if not provided.
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    
    scores = {}
    for w, results in zip(weights, result_lists):
        for rank, (product_id, _) in enumerate(results):
            scores[product_id] = scores.get(product_id, 0) + w / (k + rank + 1)
    
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return merged[:100]
