"""
SQLite store for crawled products and their price history.

Tables
  products       latest normalized record per product, plus first/last seen
  price_history  one row each time a product's price, MRP or stock changes
  crawl_runs     one row per site per crawl, for monitoring
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id          TEXT PRIMARY KEY,
    site        TEXT NOT NULL,
    data        TEXT NOT NULL,
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL,
    active      INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_products_site ON products(site);

CREATE TABLE IF NOT EXISTS price_history (
    id          TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    price       REAL NOT NULL,
    mrp         REAL,
    in_stock    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_id ON price_history(id, observed_at);

CREATE TABLE IF NOT EXISTS crawl_runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    site        TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT,
    saved       INTEGER DEFAULT 0,
    skipped     INTEGER DEFAULT 0,
    removed     INTEGER DEFAULT 0,
    note        TEXT
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CatalogStore:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------- runs
    def start_run(self, site: str) -> int:
        cur = self.db.execute("INSERT INTO crawl_runs(site, started_at, status) VALUES (?, ?, 'running')", (site, utcnow()))
        self.db.commit()
        return cur.lastrowid

    def finish_run(self, run_id: int, status: str, saved: int, skipped: int, removed: int, note: str = "") -> None:
        self.db.execute(
            "UPDATE crawl_runs SET finished_at=?, status=?, saved=?, skipped=?, removed=?, note=? WHERE run_id=?",
            (utcnow(), status, saved, skipped, removed, note, run_id),
        )
        self.db.commit()

    # --------------------------------------------------------- products
    def upsert(self, product: dict, now: str | None = None) -> tuple[bool, bool]:
        """Save a product. Returns (is_new, price_changed)."""
        now = now or utcnow()
        pid = product["_id"]
        row = self.db.execute("SELECT first_seen FROM products WHERE id=?", (pid,)).fetchone()
        is_new = row is None
        first_seen = now if is_new else row["first_seen"]
        product = {**product, "first_seen": first_seen}
        self.db.execute(
            "INSERT INTO products(id, site, data, first_seen, last_seen, active) VALUES (?, ?, ?, ?, ?, 1) "
            "ON CONFLICT(id) DO UPDATE SET data=excluded.data, last_seen=excluded.last_seen, active=1",
            (pid, product["source"], json.dumps(product, ensure_ascii=False), first_seen, now),
        )

        last = self.db.execute(
            "SELECT price, mrp, in_stock FROM price_history WHERE id=? ORDER BY observed_at DESC LIMIT 1", (pid,)
        ).fetchone()
        current = (float(product["price"]), float(product.get("mrp") or product["price"]), int(bool(product.get("in_stock"))))
        changed = last is None or (last["price"], last["mrp"], last["in_stock"]) != current
        if changed:
            self.db.execute(
                "INSERT INTO price_history(id, observed_at, price, mrp, in_stock) VALUES (?, ?, ?, ?, ?)",
                (pid, now, *current),
            )
        return is_new, (changed and last is not None and last["price"] != current[0])

    def commit(self) -> None:
        self.db.commit()

    def mark_missing_inactive(self, site: str, seen_ids: set[str]) -> int:
        """After a complete crawl, products we didn't see have been taken down."""
        rows = self.db.execute("SELECT id FROM products WHERE site=? AND active=1", (site,)).fetchall()
        gone = [r["id"] for r in rows if r["id"] not in seen_ids]
        self.db.executemany("UPDATE products SET active=0 WHERE id=?", [(g,) for g in gone])
        self.db.commit()
        return len(gone)

    # ------------------------------------------------------------ export
    def _history_summary(self, pid: str, current_price: float, days: int = 30) -> dict:
        rows = self.db.execute(
            "SELECT observed_at, price FROM price_history WHERE id=? ORDER BY observed_at", (pid,)
        ).fetchall()
        if not rows:
            return {"points": 0}
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
        window = [r["price"] for r in rows if r["observed_at"] >= cutoff] or [current_price]
        previous = None
        last_change = rows[0]["observed_at"]
        for r in reversed(rows[:-1]):
            if r["price"] != current_price:
                previous = r["price"]
                break
        for prev, nxt in zip(rows, rows[1:]):
            if prev["price"] != nxt["price"]:
                last_change = nxt["observed_at"]
        lowest = min(window + [current_price])
        return {
            "points": len(rows),
            "tracked_since": rows[0]["observed_at"],
            "previous_price": previous,
            "price_dropped": previous is not None and current_price < previous,
            "last_change": last_change,
            f"lowest_{days}d": lowest,
            f"highest_{days}d": max(window + [current_price]),
            f"is_lowest_{days}d": previous is not None and current_price <= lowest,
            "series": [[r["observed_at"][:10], r["price"]] for r in rows[-30:]],
        }

    def export(self, path: str, include_inactive: bool = False) -> int:
        query = "SELECT id, data FROM products" + ("" if include_inactive else " WHERE active=1") + " ORDER BY site, id"
        products = []
        for row in self.db.execute(query).fetchall():
            product = json.loads(row["data"])
            product["price_history"] = self._history_summary(row["id"], float(product["price"]))
            products.append(product)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(products, fh, ensure_ascii=False)
        os.replace(tmp, path)
        return len(products)

    def active_products(self) -> list[dict]:
        return [json.loads(r["data"]) for r in self.db.execute("SELECT data FROM products WHERE active=1 ORDER BY site, id")]

    # ------------------------------------------------------------- stats
    def stats(self) -> list[dict]:
        out = []
        rows = self.db.execute(
            "SELECT s.site, COUNT(p.id) AS total, COALESCE(SUM(p.active), 0) AS active, MAX(p.last_seen) AS last_seen "
            "FROM (SELECT site FROM products UNION SELECT site FROM crawl_runs) s "
            "LEFT JOIN products p ON p.site = s.site GROUP BY s.site ORDER BY s.site"
        ).fetchall()
        for row in rows:
            run = self.db.execute(
                "SELECT status, finished_at, saved, skipped, removed, note FROM crawl_runs WHERE site=? ORDER BY run_id DESC LIMIT 1",
                (row["site"],),
            ).fetchone()
            changes = self.db.execute(
                "SELECT COUNT(*) FROM price_history h JOIN products p ON p.id=h.id WHERE p.site=?", (row["site"],)
            ).fetchone()[0]
            out.append({
                "site": row["site"], "products": row["total"], "active": row["active"] or 0,
                "last_seen": row["last_seen"], "price_observations": changes,
                "last_run": dict(run) if run else None,
            })
        return out
