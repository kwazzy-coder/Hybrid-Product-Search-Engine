import importlib
import json
import os
import random
import sys
from types import SimpleNamespace

from tests.test_api import CATALOG, SERVICE, _purge, build_live_fixture


def test_metrics():
    sys.path.insert(0, SERVICE)
    from eval_metrics import judged_fraction, ndcg_at_k, recall_at_k, reciprocal_rank
    j = {"a": 3, "b": 2, "c": 0}
    assert ndcg_at_k(["a", "b", "c"], j, 3) == 1.0
    assert ndcg_at_k(["c", "b", "a"], j, 3) < 0.7
    assert ndcg_at_k(["x"], {"x": 0}, 5) is None          # no relevant items: excluded, not 0 or 1
    assert recall_at_k(["a", "x"], j, 20) == 0.5
    assert reciprocal_rank(["x", "c", "b"], j) == 1 / 3
    assert judged_fraction(["a", "x", "y", "c"], j, 4) == 0.5


class FakeClaude:
    """Stands in for anthropic.Anthropic(): labels 3 if the query's first word is in the product line."""

    def __init__(self):
        self.messages = self
        self.calls = 0

    def create(self, model, max_tokens, messages):
        self.calls += 1
        prompt = messages[0]["content"]
        query = prompt.split('Shopper\'s search: "')[1].split('"')[0]
        ratings = []
        for line in prompt.split("Products:\n")[1].splitlines():
            if line[:1].isdigit():
                n = int(line.split(".")[0])
                ratings.append({"n": n, "label": 3 if query.split()[0] in line.lower() else 0})
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps({"ratings": ratings}))])


def test_eval_pool_judge_run(tmp_path, monkeypatch):
    build_live_fixture(tmp_path, monkeypatch, EVAL_DIR=str(tmp_path / "eval"))
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "queries.txt").write_text("# test\nsilk saree\nlinen shirt\nblack tee\n")
    monkeypatch.syspath_prepend(os.path.join(SERVICE, "scripts"))
    _purge()
    ev = importlib.import_module("eval_live")

    ev.main(["pool", "--depth", "5"])
    rows = ev.read_labels()
    assert {r["query"] for r in rows} == {"silk saree", "linen shirt", "black tee"}
    assert all(r["label"] == "" for r in rows)

    fake = FakeClaude()
    ev.cmd_judge(SimpleNamespace(), client=fake)
    rows = ev.read_labels()
    assert fake.calls == 3 and all(r["label"] in {"0", "3"} for r in rows)
    assert all(r["label_source"].startswith("llm:") for r in rows)

    # A manual correction survives re-pooling
    rows[0]["label"], rows[0]["label_source"] = "2", "manual"
    ev.write_labels(rows)
    ev.main(["pool", "--depth", "5"])
    assert ev.read_labels()[0]["label"] == "2"

    ev.main(["run"])
    md = (tmp_path / "eval" / "results.md").read_text(encoding="utf-8")
    assert "| BM25 only |" in md and "| Full pipeline (as served) |" in md
    results = json.loads((tmp_path / "eval" / "results.json").read_text())
    assert results["results"]["Full pipeline (as served)"]["ndcg@10"] is not None
    _purge()


def test_retrain_on_logged_clicks(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBACK_DB", str(tmp_path / "feedback.db"))
    monkeypatch.syspath_prepend(SERVICE)
    monkeypatch.syspath_prepend(os.path.join(SERVICE, "scripts"))
    _purge()
    from feedback_db import log_click_event, log_impressions

    # Simulated logs where shoppers click results that match the category (feature 1),
    # regardless of where they were shown. The RRF score (feature 0) is noise.
    rng = random.Random(0)
    for s in range(60):
        rows = []
        for pos in range(10):
            fv = [rng.random() * 0.03, float(rng.random() < 0.3)] + [rng.random() for _ in range(10)]
            rows.append((pos, f"p{s}_{pos}", fv))
        qid = f"q{s}"
        log_impressions(qid, "linen shirt", "text", rows, feature_version=1, model_version="test")
        matches = [r for r in rows if r[2][1] == 1.0 and r[0] > 0]
        if matches:
            pos, pid, _ = matches[0]
            log_click_event("linen shirt", [r[1] for r in rows], pid, query_id=qid, position=pos)

    train = importlib.import_module("train_ltr_live")
    groups = train.build_groups(train.get_training_sessions(1))
    assert groups and all(set(g["y"]) == {0, 1} for g in groups)

    out = tmp_path / "ltr_model.lgb"
    monkeypatch.setattr(sys, "argv", ["train_ltr_live.py", "--min-sessions", "10", "--out", str(out), "--force"])
    assert train.main() == 0
    report = json.loads((tmp_path / "ltr_model_report.json").read_text())
    holdout = report["holdout"]
    assert holdout["New model"]["ndcg@10"] > holdout["RRF score only"]["ndcg@10"]
    assert report["feature_importance_gain"]["category_match"] == max(report["feature_importance_gain"].values())
    _purge()
