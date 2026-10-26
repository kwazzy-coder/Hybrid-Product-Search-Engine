"""
Offline evaluation on your real (crawled) catalog.

    cd search_service
    python scripts/eval_live.py pool             # 1. collect candidate results for each test query
    python scripts/eval_live.py judge            # 2. label them 0-3 with Claude (needs ANTHROPIC_API_KEY)
                                                 #    ...or open data/eval/labels.csv in Excel and fill the label column
    python scripts/eval_live.py run              # 3. compute metrics for every stage of the pipeline

Queries live in data/eval/queries.txt (one per line, edit freely).
Labels live in data/eval/labels.csv. Spot-check the LLM's labels and fix any you
disagree with (set label_source to "manual" on rows you change).
Results are written to data/eval/results.md, ready to paste into the README.

Each query's candidates are "pooled": the union of the top results from every
variant. Unlabelled results count as not relevant; the judged@10 column shows
how much of each variant's top 10 was actually labelled, so you can tell whether
a number can be trusted.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

SERVICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SERVICE_DIR)

from eval_metrics import judged_fraction, mean, ndcg_at_k, percentile, recall_at_k, reciprocal_rank  # noqa: E402
from paths import CATALOG_MODE, CATALOG_PATH, MATCHES_PATH, PROJECT_DIR  # noqa: E402

EVAL_DIR = os.getenv("EVAL_DIR", os.path.join(PROJECT_DIR, "data", "eval"))
QUERIES_PATH = os.path.join(EVAL_DIR, "queries.txt")
LABELS_PATH = os.path.join(EVAL_DIR, "labels.csv")
RESULTS_MD = os.path.join(EVAL_DIR, "results.md")
RESULTS_JSON = os.path.join(EVAL_DIR, "results.json")
CSV_FIELDS = ["query", "product_id", "store", "name", "category", "colors", "price", "in_stock", "url",
              "label", "label_source"]
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "claude-sonnet-5-5")


# ------------------------------------------------------------------ context
class SearchContext:
    """Loads the catalog, indexes and model the same way the API does."""

    def __init__(self):
        from deduplication import build_dedup_index
        from reranker.ltr_model import LTRModel
        from retrieval.bm25_retriever import BM25Retriever
        from retrieval.clip_retriever import CLIPRetriever
        from retrieval.faiss_retriever import FAISSRetriever

        if not CATALOG_PATH or not os.path.exists(CATALOG_PATH):
            sys.exit("No catalog found. Run `python -m crawler refresh` and scripts/build_live_index.py first.")
        with open(CATALOG_PATH, encoding="utf-8") as fh:
            self.products = json.load(fh)
        self.product_map = {p["_id"]: p for p in self.products}
        self.bm25, self.faiss, self.clip = BM25Retriever(), FAISSRetriever(), CLIPRetriever()
        if not (self.bm25.load() and self.faiss.load()):
            sys.exit("Search indexes not found. Run scripts/build_live_index.py first.")
        self.clip.load()
        self.ltr = LTRModel()
        groups = {}
        if os.path.exists(MATCHES_PATH):
            with open(MATCHES_PATH, encoding="utf-8") as fh:
                groups = json.load(fh)
        self.dedup_index = build_dedup_index(self.products, groups)
        self.categories = {str(p.get("category", "")).lower() for p in self.products if p.get("category")}
        print(f"Loaded {len(self.products)} products ({CATALOG_MODE} catalog), LTR model {self.ltr.version}")

    def variants(self):
        """name -> function(intent) returning ranked product ids."""
        from pipeline import Filters, fuse, rank, resolve_filters
        from query_understanding import KNOWN_CATEGORIES

        known = KNOWN_CATEGORIES | self.categories

        def ids(results):
            return [pid for pid, _ in results]

        def hybrid_parts(intent):
            b = self.bm25.search(intent.expanded_terms)
            f = self.faiss.search(intent.original_query)
            c = self.clip.search(intent.original_query)
            return b, f, c

        def full(intent, **switches):
            b, f, c = hybrid_parts(intent)
            fused = fuse(b, f, c)
            filters = resolve_filters(intent, Filters(), known)
            ranked = rank(intent, fused, self.product_map, dict(c), self.ltr, self.dedup_index, filters, **switches)
            return [cand.product_id for cand in ranked]

        out = {
            "BM25 only": lambda intent: ids(self.bm25.search(intent.expanded_terms)),
            "MiniLM vectors only": lambda intent: ids(self.faiss.search(intent.original_query)),
        }
        if self.clip.index.ntotal > 0:
            out["CLIP text-to-image only"] = lambda intent: ids(self.clip.search(intent.original_query))
        out["Hybrid (RRF fusion)"] = lambda intent: ids(fuse(*hybrid_parts(intent)))
        out["Hybrid + LTR re-ranking"] = lambda intent: full(intent, use_diversity=False, use_filters=False)
        out["Full pipeline (as served)"] = lambda intent: full(intent)
        return out


def read_queries():
    if not os.path.exists(QUERIES_PATH):
        sys.exit(f"Missing {QUERIES_PATH}")
    with open(QUERIES_PATH, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]


def read_labels():
    if not os.path.exists(LABELS_PATH):
        return []
    with open(LABELS_PATH, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_labels(rows):
    os.makedirs(EVAL_DIR, exist_ok=True)
    tmp = LABELS_PATH + ".tmp"
    # utf-8-sig so Excel on Windows shows ₹ and non-English names correctly
    with open(tmp, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, LABELS_PATH)


def parse_intents(queries):
    from query_understanding import parse_query
    return {q: parse_query(q) for q in queries}


# ---------------------------------------------------------------------- pool
def cmd_pool(args):
    ctx = SearchContext()
    queries = read_queries()
    intents = parse_intents(queries)
    variants = ctx.variants()
    existing = read_labels()
    have = {(r["query"], r["product_id"]) for r in existing}
    added = 0
    rows = list(existing)
    for q in queries:
        pooled = []
        for fn in variants.values():
            for pid in fn(intents[q])[:args.depth]:
                if pid not in pooled:
                    pooled.append(pid)
        for pid in pooled:
            if (q, pid) in have or pid not in ctx.product_map:
                continue
            p = ctx.product_map[pid]
            rows.append({
                "query": q, "product_id": pid, "store": p.get("source_name") or p.get("source", ""),
                "name": p.get("name", ""), "category": p.get("category", ""),
                "colors": ", ".join(p.get("color") or []), "price": f"{float(p.get('price') or 0):g}",
                "in_stock": "yes" if p.get("in_stock", True) else "no", "url": p.get("source_url", ""),
                "label": "", "label_source": "",
            })
            added += 1
    rows.sort(key=lambda r: (queries.index(r["query"]) if r["query"] in queries else 10**6, r["store"], r["name"]))
    write_labels(rows)
    unlabeled = sum(1 for r in rows if r["label"] == "")
    print(f"{len(queries)} queries, {added} new candidates added, {unlabeled} still need a label -> {LABELS_PATH}")


# --------------------------------------------------------------------- judge
JUDGE_PROMPT = """You are rating search results for an Indian fashion shopping site.

Shopper's search: "{query}"

Rate how well each product matches that search:
3 = exactly what was asked: right product type and every stated attribute (colour, material, fit, size, price limit, occasion)
2 = right product type, but misses or only partly matches a stated attribute
1 = related but a different product type (a substitute or something worn with it)
0 = not relevant

Products:
{items}

Reply with JSON only, no other text: {{"ratings": [{{"n": 1, "label": 3}}, ...]}} with one entry per product."""


def judge_batch(client, query, batch):
    items = "\n".join(
        f"{i}. [{r['category']}] {r['name']} | colours: {r['colors'] or 'n/a'} | ₹{r['price']} | {r['store']}"
        + ("" if r["in_stock"] == "yes" else " | out of stock")
        for i, r in enumerate(batch, 1)
    )
    msg = client.messages.create(
        model=JUDGE_MODEL, max_tokens=1500,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(query=query, items=items)}],
    )
    text = "".join(getattr(block, "text", "") for block in msg.content)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    data = json.loads(match.group(0) if match else text)
    return {int(r["n"]): int(r["label"]) for r in data.get("ratings", []) if 0 <= int(r["label"]) <= 3}


def cmd_judge(args, client=None):
    rows = read_labels()
    if not rows:
        sys.exit("No candidates yet. Run `python scripts/eval_live.py pool` first.")
    if client is None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY to label with Claude, or fill the label column in "
                     f"{LABELS_PATH} yourself (0-3) and skip this step.")
        import anthropic
        client = anthropic.Anthropic()
    todo = defaultdict(list)
    for r in rows:
        if r["label"] == "":
            todo[r["query"]].append(r)
    print(f"Labelling {sum(len(v) for v in todo.values())} results for {len(todo)} queries with {JUDGE_MODEL} ...")
    done = 0
    for query, items in todo.items():
        for start in range(0, len(items), 40):
            batch = items[start:start + 40]
            try:
                labels = judge_batch(client, query, batch)
            except Exception as exc:
                print(f"  {query!r}: failed ({exc}); leaving those unlabelled")
                continue
            for i, r in enumerate(batch, 1):
                if i in labels:
                    r["label"], r["label_source"] = str(labels[i]), f"llm:{JUDGE_MODEL}"
                    done += 1
        write_labels(rows)   # save as we go
    print(f"Labelled {done}. Open {LABELS_PATH}, spot-check, and fix anything you disagree with.")


# ----------------------------------------------------------------------- run
def cmd_run(args):
    rows = [r for r in read_labels() if r["label"] != ""]
    if not rows:
        sys.exit("No labels yet. Run `pool` then `judge` (or label by hand) first.")
    judgments = defaultdict(dict)
    sources = defaultdict(int)
    for r in rows:
        judgments[r["query"]][r["product_id"]] = int(r["label"])
        sources["manual" if not r["label_source"].startswith("llm") else "llm"] += 1
    queries = [q for q in read_queries() if q in judgments]

    ctx = SearchContext()
    intents = parse_intents(queries)
    results = {}
    for name, fn in ctx.variants().items():
        fn(intents[queries[0]])                     # warm-up (model loading)
        per_query, latencies = [], []
        for q in queries:
            t0 = time.perf_counter()
            ranked = fn(intents[q])
            latencies.append((time.perf_counter() - t0) * 1000)
            j = judgments[q]
            per_query.append({
                "query": q,
                "ndcg@5": ndcg_at_k(ranked, j, 5), "ndcg@10": ndcg_at_k(ranked, j, 10),
                "recall@20": recall_at_k(ranked, j, 20), "mrr": reciprocal_rank(ranked[:20], j),
                "judged@10": judged_fraction(ranked, j, 10),
            })
        results[name] = {
            **{m: mean(pq[m] for pq in per_query) for m in ("ndcg@5", "ndcg@10", "recall@20", "mrr", "judged@10")},
            "p50_ms": percentile(latencies, 50), "p95_ms": percentile(latencies, 95), "per_query": per_query,
        }

    def fmt(v, pct=False):
        return "n/a" if v is None else (f"{v * 100:.0f}%" if pct else f"{v:.3f}")

    lines = [
        f"## Search quality on the live catalog ({len(ctx.products)} products, {len(queries)} queries)",
        "",
        "| Variant | NDCG@5 | NDCG@10 | Recall@20 | MRR | judged@10 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in results.items():
        lines.append(f"| {name} | {fmt(m['ndcg@5'])} | {fmt(m['ndcg@10'])} | {fmt(m['recall@20'])} | {fmt(m['mrr'])} | "
                     f"{fmt(m['judged@10'], pct=True)} | {m['p50_ms']:.0f} | {m['p95_ms']:.0f} |")
    lines += [
        "",
        f"Labels: {sum(sources.values())} graded 0-3 ({sources['llm']} by {JUDGE_MODEL}, spot-checked; "
        f"{sources['manual']} by hand). Relevant = label ≥ 2 for Recall and MRR. Candidates pooled from every variant's "
        "top results; unlabelled results count as not relevant. Latency covers retrieval and ranking, not query "
        f"parsing, on the machine that ran the evaluation. Generated {datetime.now(timezone.utc):%Y-%m-%d}.",
    ]
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(RESULTS_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(RESULTS_JSON, "w", encoding="utf-8") as fh:
        json.dump({"queries": queries, "results": results, "label_sources": dict(sources)}, fh, indent=1)
    print("\n".join(lines))
    print(f"\nSaved {RESULTS_MD}")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool", help="collect candidates to label")
    p.add_argument("--depth", type=int, default=10, help="top-N from each variant (10 keeps hand-labelling manageable)")
    sub.add_parser("judge", help="label unlabelled candidates with Claude")
    sub.add_parser("run", help="compute metrics")
    args = parser.parse_args(argv)
    return {"pool": cmd_pool, "judge": cmd_judge, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
