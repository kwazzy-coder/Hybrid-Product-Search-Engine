import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "search_service"))

from deduplication import match_cross_store, build_dedup_index  # noqa: E402


def vec(seed, noise=0.0, base=None):
    rng = np.random.default_rng(seed)
    v = base.copy() if base is not None else rng.standard_normal(512)
    if noise:  # noise is relative to a unit vector, so cosine ≈ 1 / sqrt(1 + noise²)
        v = v / np.linalg.norm(v) + rng.standard_normal(512) * noise / np.sqrt(512)
    return (v / np.linalg.norm(v)).astype("float32")


def test_same_product_on_two_stores_is_matched_and_lookalikes_are_not():
    shirt = vec(1)
    products = [
        {"_id": "a_1", "source": "a", "brand": "Libas", "category": "kurta", "name": "Libas Yellow Cotton Kurta", "price": 1299},
        {"_id": "b_9", "source": "b", "brand": "Libas", "category": "kurta", "name": "Yellow Cotton Straight Kurta", "price": 1199},
        # Looks almost the same but different store AND different brand AND unrelated title -> no match
        {"_id": "c_3", "source": "c", "brand": "Other", "category": "kurta", "name": "Mustard Anarkali", "price": 999},
        # Same store as a_1 -> never matched (colour variants are listed separately)
        {"_id": "a_2", "source": "a", "brand": "Libas", "category": "kurta", "name": "Libas Yellow Cotton Kurta", "price": 1299},
        # Different category -> no match even with an identical image
        {"_id": "d_4", "source": "d", "brand": "Libas", "category": "dress", "name": "Libas Yellow Cotton Kurta", "price": 1299},
    ]
    vectors = {
        "a_1": shirt, "b_9": vec(2, noise=0.15, base=shirt), "c_3": vec(3, noise=0.15, base=shirt),
        "a_2": vec(4, noise=0.05, base=shirt), "d_4": shirt,
    }
    groups = match_cross_store(products, vectors)
    assert list(groups.values()) == [["a_1", "a_2", "b_9"]] or list(groups.values()) == [["a_1", "b_9"]]
    members = next(iter(groups.values()))
    assert "c_3" not in members and "d_4" not in members

    index = build_dedup_index(products, groups)
    assert index["b_9"]["price_rank"] == 1 and index["b_9"]["cheapest_price"] == 1199
    assert index["a_1"]["cheapest_id"] == "b_9"


def test_no_vectors_no_matches():
    assert match_cross_store([{"_id": "x", "source": "a"}], {}) == {}
