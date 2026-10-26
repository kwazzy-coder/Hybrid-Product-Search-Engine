"""
SQLite store for user feedback: what was shown (impressions), what was clicked,
and price-drop alerts.

Impressions record the exact feature vector each result had when it was shown,
so the re-ranker can be retrained on real behaviour with the real features
(scripts/train_ltr_live.py) instead of made-up ones.
"""

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional


def get_db_path() -> str:
    """Absolute path to feedback.db. FEEDBACK_DB overrides (e.g. when data/ is read-only)."""
    override = os.getenv("FEEDBACK_DB")
    if override:
        os.makedirs(os.path.dirname(os.path.abspath(override)), exist_ok=True)
        return override
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_data_dir = os.path.join(os.path.dirname(current_dir), "data")
    if os.path.exists(parent_data_dir):
        return os.path.join(parent_data_dir, "feedback.db")
    local_data_dir = os.path.join(current_dir, "data")
    os.makedirs(local_data_dir, exist_ok=True)
    return os.path.join(local_data_dir, "feedback.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path(), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if needed and migrate older databases in place."""
    db_path = get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS click_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            shown_results TEXT NOT NULL,
            clicked_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS impressions (
            query_id TEXT NOT NULL,
            query TEXT NOT NULL,
            mode TEXT,
            position INTEGER NOT NULL,
            product_id TEXT NOT NULL,
            features TEXT NOT NULL,
            feature_version INTEGER NOT NULL,
            model_version TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_impressions_query ON impressions(query_id);
        CREATE TABLE IF NOT EXISTS price_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id TEXT NOT NULL,
            product_name TEXT,
            source_name TEXT,
            source_url TEXT,
            email TEXT NOT NULL,
            target_price REAL NOT NULL,
            price_when_set REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            active INTEGER NOT NULL DEFAULT 1,
            notified_at DATETIME,
            notified_price REAL
        );
        CREATE INDEX IF NOT EXISTS idx_alerts_active ON price_alerts(active);
    """)
    # Older click_events tables lack query_id/position.
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(click_events)")}
    if "query_id" not in columns:
        conn.execute("ALTER TABLE click_events ADD COLUMN query_id TEXT")
    if "position" not in columns:
        conn.execute("ALTER TABLE click_events ADD COLUMN position INTEGER")
    conn.commit()
    conn.close()


# ------------------------------------------------------------------ clicks
def log_click_event(query: str, shown_results: List[str], clicked_id: str, timestamp: Optional[str] = None,
                    query_id: Optional[str] = None, position: Optional[int] = None) -> int:
    init_db()
    conn = _connect()
    cols = ["query", "shown_results", "clicked_id", "query_id", "position"]
    vals = [query, json.dumps(shown_results or []), clicked_id, query_id, position]
    if timestamp:
        cols.append("timestamp")
        vals.append(timestamp)
    cur = conn.execute(f"INSERT INTO click_events ({', '.join(cols)}) VALUES ({', '.join('?' * len(vals))})", vals)
    conn.commit()
    event_id = cur.lastrowid
    conn.close()
    return event_id


def get_all_click_events() -> List[Dict[str, Any]]:
    init_db()
    conn = _connect()
    rows = conn.execute("SELECT id, query, shown_results, clicked_id, timestamp, query_id, position FROM click_events").fetchall()
    conn.close()
    return [{
        "id": r["id"], "query": r["query"], "shown_results": json.loads(r["shown_results"] or "[]"),
        "clicked_id": r["clicked_id"], "timestamp": r["timestamp"], "query_id": r["query_id"], "position": r["position"],
    } for r in rows]


# ------------------------------------------------------------- impressions
def log_impressions(query_id: str, query: str, mode: str, rows: List[tuple], feature_version: int,
                    model_version: Optional[str] = None) -> None:
    """rows: (position, product_id, feature_vector). Called in the background after each search."""
    if not rows:
        return
    try:
        init_db()
        conn = _connect()
        conn.executemany(
            "INSERT INTO impressions (query_id, query, mode, position, product_id, features, feature_version, model_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(query_id, query, mode, int(pos), pid, json.dumps([round(float(x), 6) for x in fv]), feature_version, model_version)
             for pos, pid, fv in rows],
        )
        conn.commit()
        conn.close()
    except Exception as exc:  # logging must never break search
        print(f"Impression logging failed (non-critical): {exc}")


def get_training_sessions(feature_version: int) -> List[Dict[str, Any]]:
    """
    One session per searched query that got at least one click:
    {"query_id", "query", "timestamp", "items": [{"position", "product_id", "features", "clicked"}]}
    """
    init_db()
    conn = _connect()
    clicks: Dict[str, set] = {}
    for r in conn.execute("SELECT query_id, clicked_id FROM click_events WHERE query_id IS NOT NULL"):
        clicks.setdefault(r["query_id"], set()).add(r["clicked_id"])
    sessions = []
    for qid, clicked in clicks.items():
        rows = conn.execute(
            # Text searches only: photo results are ranked by visual similarity, not by these features.
            "SELECT query, position, product_id, features, timestamp FROM impressions "
            "WHERE query_id=? AND feature_version=? AND (mode IS NULL OR mode='text') ORDER BY position",
            (qid, feature_version)
        ).fetchall()
        if not rows:
            continue
        sessions.append({
            "query_id": qid, "query": rows[0]["query"], "timestamp": rows[0]["timestamp"],
            "items": [{"position": r["position"], "product_id": r["product_id"],
                       "features": json.loads(r["features"]), "clicked": r["product_id"] in clicked} for r in rows],
        })
    conn.close()
    sessions.sort(key=lambda s: s["timestamp"] or "")
    return sessions


# ------------------------------------------------------------------ alerts
def create_alert(product: dict, email: str, target_price: float) -> int:
    init_db()
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO price_alerts (product_id, product_name, source_name, source_url, email, target_price, price_when_set) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (product["_id"], product.get("name"), product.get("source_name") or product.get("source"),
         product.get("source_url"), email.strip().lower(), float(target_price), float(product.get("price") or 0)),
    )
    conn.commit()
    alert_id = cur.lastrowid
    conn.close()
    return alert_id


def count_active_alerts(email: str) -> int:
    init_db()
    conn = _connect()
    n = conn.execute("SELECT COUNT(*) FROM price_alerts WHERE email=? AND active=1", (email.strip().lower(),)).fetchone()[0]
    conn.close()
    return n
