"""Evaluate the running search API using reproducible relevance judgments.

Run: python scripts/evaluate_search.py --base-url http://localhost:8000
"""
import argparse
import json
from math import log2
from urllib.parse import urlencode
from urllib.request import urlopen

JUDGMENTS = [
    {"query": "red shirt", "category": "shirt", "color": "red"},
    {"query": "blue jeans", "category": "jeans", "color": "blue"},
    {"query": "pink dress", "category": "dress", "color": "pink"},
    {"query": "white tshirt", "category": "tshirt", "color": "white"},
]


def relevant(product, judgment):
    product_text = " ".join([
        str(product.get("category", "")), str(product.get("name", "")),
        " ".join(product.get("tags", [])), " ".join(product.get("color", [])),
    ]).lower()
    return judgment["category"] in product_text and judgment["color"] in product_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    recalls, reciprocal_ranks, ndcgs = [], [], []
    for judgment in JUDGMENTS:
        url = f"{args.base_url}/search?{urlencode({'q': judgment['query'], 'limit': args.k})}"
        with urlopen(url, timeout=30) as response:
            results = json.load(response)["results"]
        relevance = [relevant(product, judgment) for product in results]
        hits = sum(relevance)
        recalls.append(1.0 if hits else 0.0)
        first_rank = next((index + 1 for index, hit in enumerate(relevance) if hit), None)
        reciprocal_ranks.append(1 / first_rank if first_rank else 0.0)
        dcg = sum(1 / log2(index + 2) for index, hit in enumerate(relevance) if hit)
        ideal_dcg = sum(1 / log2(index + 2) for index in range(min(hits, args.k)))
        ndcgs.append(dcg / ideal_dcg if ideal_dcg else 0.0)
    print(json.dumps({
        "queries": len(JUDGMENTS), "k": args.k,
        "Recall@K": round(sum(recalls) / len(recalls), 4),
        "MRR": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4),
        "NDCG@K": round(sum(ndcgs) / len(ndcgs), 4),
    }, indent=2))


if __name__ == "__main__":
    main()
