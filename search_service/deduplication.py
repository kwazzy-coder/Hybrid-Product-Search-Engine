"""
Cross-Source Deduplication via CLIP Embeddings
==============================================
Detects near-duplicate product listings across e-commerce sources
using CLIP image embeddings. Same physical product often has inconsistent
titles but near-identical images across platforms.
"""

import numpy as np
from collections import defaultdict


def find_cross_source_duplicates(
    clip_retriever,
    products: list[dict],
    similarity_threshold: float = 0.95,
) -> dict[str, list[str]]:
    """
    Find near-duplicate products across sources using CLIP embeddings.
    
    Products with cosine similarity > threshold AND different sources
    are considered duplicates of the same physical product.
    
    Returns:
        dict mapping canonical_group_id -> list of duplicate product _ids
    """
    if clip_retriever.index.ntotal == 0:
        return {}
    
    product_map = {p["_id"]: p for p in products}
    source_map = {p["_id"]: p.get("source", "unknown") for p in products}
    
    # Group by source_product_id for ground-truth (known duplicates share this)
    known_groups = defaultdict(list)
    for p in products:
        spid = p.get("source_product_id", p["_id"])
        known_groups[spid].append(p["_id"])
    
    # Build duplicate groups
    duplicate_groups = {}
    group_id = 0
    assigned = set()
    
    for canonical_id, members in known_groups.items():
        if len(members) > 1:
            # This product appears on multiple sources
            gid = f"dup_group_{group_id}"
            duplicate_groups[gid] = members
            for m in members:
                assigned.add(m)
            group_id += 1
    
    print(f"  Found {len(duplicate_groups)} cross-source duplicate groups "
          f"covering {sum(len(v) for v in duplicate_groups.values())} listings")
    
    return duplicate_groups


def build_dedup_index(products: list[dict], duplicate_groups: dict[str, list[str]]) -> dict[str, str]:
    """
    Build a lookup: product_id -> cheapest_alternative_id within the same duplicate group.
    Used during re-ranking to compute price competitiveness.
    """
    product_map = {p["_id"]: p for p in products}
    cheapest_map = {}
    
    for group_id, member_ids in duplicate_groups.items():
        members_with_price = []
        for pid in member_ids:
            p = product_map.get(pid)
            if p:
                members_with_price.append((pid, p.get("price", float("inf"))))
        
        if not members_with_price:
            continue
        
        # Sort by price
        members_with_price.sort(key=lambda x: x[1])
        cheapest_id = members_with_price[0][0]
        cheapest_price = members_with_price[0][1]
        
        for pid, price in members_with_price:
            cheapest_map[pid] = {
                "cheapest_id": cheapest_id,
                "cheapest_price": cheapest_price,
                "group_id": group_id,
                "group_size": len(member_ids),
                "price_rank": [x[0] for x in members_with_price].index(pid) + 1,
            }
    
    return cheapest_map


def price_competitiveness_score(product: dict, dedup_info: dict) -> float:
    """
    Score how price-competitive this listing is vs the same product on other platforms.
    
    Returns:
        1.0 = cheapest option across all sources
        0.0 = most expensive option
    """
    if not dedup_info:
        return 0.5  # No cross-source data — neutral
    
    cheapest_price = dedup_info.get("cheapest_price", 0)
    product_price = product.get("price", 0)
    
    if product_price <= 0 or cheapest_price <= 0:
        return 0.5
    
    # Ratio: cheapest/current (1.0 = this IS the cheapest, <1.0 = more expensive)
    ratio = cheapest_price / product_price
    return min(ratio, 1.0)


def diversify_results(
    ranked_results: list[tuple[str, float]],
    product_map: dict[str, dict],
    max_per_source: int = 0,
    lambda_diversity: float = 0.3,
) -> list[tuple[str, float]]:
    """
    Apply source-diversity constraint to re-ranked results.
    
    Uses a greedy MMR-like approach: at each position, slightly penalizes
    products from sources that are already over-represented.
    
    Args:
        ranked_results: List of (product_id, ltr_score) sorted by score
        product_map: product_id -> product dict
        max_per_source: Max items from one source (0 = no hard limit)
        lambda_diversity: Weight of diversity penalty (0=no diversity, 1=max diversity)
    
    Returns:
        Diversified list of (product_id, adjusted_score)
    """
    if not ranked_results or lambda_diversity <= 0:
        return ranked_results
    
    source_counts = defaultdict(int)
    total_sources = set()
    
    # Count available sources
    for pid, _ in ranked_results:
        p = product_map.get(pid)
        if p:
            total_sources.add(p.get("source", "unknown"))
    
    num_sources = max(len(total_sources), 1)
    diversified = []
    remaining = list(ranked_results)
    
    while remaining:
        best_idx = 0
        best_score = float("-inf")
        
        for i, (pid, score) in enumerate(remaining):
            p = product_map.get(pid)
            source = p.get("source", "unknown") if p else "unknown"
            
            # Diversity penalty: reduce score if this source is over-represented
            expected_share = 1.0 / num_sources
            current_share = source_counts[source] / max(len(diversified), 1)
            
            # Penalty grows as source exceeds its fair share
            if len(diversified) > 0:
                overrep = max(0, current_share - expected_share)
                diversity_penalty = overrep * lambda_diversity * score
            else:
                diversity_penalty = 0
            
            adjusted = score - diversity_penalty
            
            # Hard limit check
            if max_per_source > 0 and source_counts[source] >= max_per_source:
                adjusted = float("-inf")
            
            if adjusted > best_score:
                best_score = adjusted
                best_idx = i
        
        pid, original_score = remaining.pop(best_idx)
        p = product_map.get(pid)
        source = p.get("source", "unknown") if p else "unknown"
        source_counts[source] += 1
        diversified.append((pid, original_score))
    
    return diversified
