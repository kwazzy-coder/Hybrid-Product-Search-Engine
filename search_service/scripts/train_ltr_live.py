"""
Retrain the LightGBM re-ranker on real clicks from the website.

Every search logs what was shown and the exact 12 features each result had at that
moment (the `impressions` table); every click is logged with its position. Each
search that got a click becomes one training group:

    clicked results                          -> label 1
    results shown above a click, not clicked -> label 0   ("skip-above": the user saw them and passed)
    the result right below the lowest click  -> label 0   (very likely seen too)
    anything further down                    -> dropped   (the user probably never looked)

Dropping the unexamined tail reduces position bias: people click top results
partly because they're on top, not only because they're better.

The newest 20% of sessions are held out. The new model is saved only if it beats
the current one on them (use --force to save anyway).

    cd search_service
    python scripts/train_ltr_live.py              # needs ~30+ searches with clicks
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

SERVICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SERVICE_DIR)

import numpy as np  # noqa: E402

from eval_metrics import mean, ndcg_at_k  # noqa: E402
from feedback_db import get_training_sessions  # noqa: E402
from paths import INDEX_DIR  # noqa: E402
from reranker.features import FEATURE_VERSION  # noqa: E402

FEATURE_NAMES = ["rrf", "category_match", "color_match", "price_match", "rating", "reviews", "discount",
                 "in_stock", "clip_score", "attribute_match", "binding", "price_competitiveness"]


def build_groups(sessions):
    """Turn logged sessions into (features, labels) groups with skip-above labelling."""
    groups = []
    for s in sessions:
        items = sorted(s["items"], key=lambda it: it["position"])
        clicked_positions = [it["position"] for it in items if it["clicked"]]
        if not clicked_positions:
            continue
        cutoff = max(clicked_positions) + 1
        kept = [it for it in items if it["position"] <= cutoff]
        labels = [1 if it["clicked"] else 0 for it in kept]
        if len(set(labels)) < 2:      # need at least one positive and one negative to learn anything
            continue
        groups.append({"query": s["query"], "X": [it["features"] for it in kept], "y": labels})
    return groups


def evaluate(groups, score_fn) -> dict:
    n5, n10 = [], []
    for g in groups:
        scores = score_fn(np.array(g["X"], dtype=np.float32))
        order = list(np.argsort(-np.asarray(scores), kind="stable"))
        ranked = [str(i) for i in order]
        judgments = {str(i): label for i, label in enumerate(g["y"])}
        n5.append(ndcg_at_k(ranked, judgments, 5))
        n10.append(ndcg_at_k(ranked, judgments, 10))
    return {"ndcg@5": mean(n5), "ndcg@10": mean(n10)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-sessions", type=int, default=30)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--out", default=os.path.join(INDEX_DIR, "ltr_model.lgb"))
    parser.add_argument("--force", action="store_true", help="save even if it doesn't beat the current model")
    args = parser.parse_args()

    import lightgbm as lgb
    from reranker.ltr_model import LTRModel

    sessions = get_training_sessions(FEATURE_VERSION)
    groups = build_groups(sessions)
    print(f"{len(sessions)} searches with clicks, {len(groups)} usable for training "
          f"(need a click below at least one unclicked result).")
    if len(groups) < args.min_sessions and not args.force:
        print(f"Not enough data yet: collect at least {args.min_sessions} usable searches by using the site, "
              f"then run this again (or pass --force to train anyway).")
        return 1

    n_valid = max(1, int(len(groups) * args.holdout)) if len(groups) >= 5 else 0
    train, valid = groups[:len(groups) - n_valid], groups[len(groups) - n_valid:]
    X = np.array([x for g in train for x in g["X"]], dtype=np.float32)
    y = np.array([label for g in train for label in g["y"]], dtype=np.float32)
    params = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5, 10], "num_leaves": 15,
              "min_data_in_leaf": 5, "learning_rate": 0.05, "verbose": -1}
    dtrain = lgb.Dataset(X, label=y, group=[len(g["y"]) for g in train])
    callbacks = []
    valid_sets = []
    if valid:
        Xv = np.array([x for g in valid for x in g["X"]], dtype=np.float32)
        yv = np.array([label for g in valid for label in g["y"]], dtype=np.float32)
        valid_sets = [lgb.Dataset(Xv, label=yv, group=[len(g["y"]) for g in valid], reference=dtrain)]
        callbacks = [lgb.early_stopping(20, verbose=False)]
    model = lgb.train(params, dtrain, num_boost_round=300, valid_sets=valid_sets, callbacks=callbacks)

    report = {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "train_groups": len(train), "holdout_groups": len(valid), "feature_version": FEATURE_VERSION}
    if valid:
        current = LTRModel()
        rows = {
            "Order users saw (logged)": evaluate(valid, lambda Xg: -np.arange(len(Xg))),
            "RRF score only": evaluate(valid, lambda Xg: Xg[:, 0]),
            f"Current model ({current.version})": evaluate(valid, lambda Xg: current.predict(Xg.tolist())),
            "New model": evaluate(valid, model.predict),
        }
        report["holdout"] = rows
        print(f"\nHeld-out NDCG on the newest {len(valid)} searches:")
        for name, m in rows.items():
            print(f"  {name:<45} NDCG@5 {m['ndcg@5']:.3f}   NDCG@10 {m['ndcg@10']:.3f}")
        better = rows["New model"]["ndcg@10"] > list(rows.values())[2]["ndcg@10"]
    else:
        better = False
        print("Too few sessions for a holdout; can't compare against the current model.")

    importance = dict(zip(FEATURE_NAMES, model.feature_importance(importance_type="gain").round(1).tolist()))
    report["feature_importance_gain"] = importance
    print("\nWhat the new model relies on (gain):")
    for name, gain in sorted(importance.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {name:<22} {gain}")

    if better or args.force:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        model.save_model(args.out)
        with open(os.path.splitext(args.out)[0] + "_report.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nSaved to {args.out}. Restart the API (or POST /admin/reload_ltr) to use it.")
    else:
        print("\nNot saved: the new model didn't beat the current one on held-out searches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
