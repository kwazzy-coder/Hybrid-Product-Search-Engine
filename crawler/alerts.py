"""
Price-drop alerts: after each crawl, email everyone whose target price has been reached.

Alerts are created from the website (POST /api/alerts) and stored in data/feedback.db.
Email goes out over SMTP when these environment variables are set:

    SMTP_HOST      e.g. smtp.gmail.com
    SMTP_PORT      default 587 (STARTTLS)
    SMTP_USER      the account to log in with
    SMTP_PASSWORD  for Gmail, an "app password", not your normal password
    ALERT_FROM     optional sender address (defaults to SMTP_USER)

Without SMTP settings, notifications are printed and appended to
data/alerts_outbox.txt, so you can see them working before you set up email.
Each alert fires once and is then switched off.
"""

from __future__ import annotations

import json
import os
import smtplib
import sqlite3
from datetime import datetime, timezone
from email.message import EmailMessage

from .store import CatalogStore


def _smtp_settings() -> dict | None:
    host = os.getenv("SMTP_HOST")
    if not host:
        return None
    user = os.getenv("SMTP_USER", "")
    return {
        "host": host, "port": int(os.getenv("SMTP_PORT", "587")), "user": user,
        "password": os.getenv("SMTP_PASSWORD", ""), "sender": os.getenv("ALERT_FROM") or user,
    }


def _message(alert: sqlite3.Row, product: dict) -> tuple[str, str]:
    price, target = float(product["price"]), float(alert["target_price"])
    store = product.get("source_name") or alert["source_name"] or "the store"
    subject = f"Price drop: {product.get('name')} is now ₹{price:,.0f}"
    body = (
        f"Good news! {product.get('name')} from {store} is now ₹{price:,.0f}.\n\n"
        f"You asked to hear when it dropped to ₹{target:,.0f} or less"
        + (f" (it was ₹{float(alert['price_when_set']):,.0f} when you set the alert)" if alert["price_when_set"] else "")
        + f".\n\nView it on {store}: {product.get('source_url') or alert['source_url']}\n\n"
        "This was a one-time alert, so you won't hear about this product again unless you set a new one.\n\n"
        "Dhaaga\n"
    )
    return subject, body


def _deliver(email: str, subject: str, body: str, smtp: dict | None, outbox_path: str) -> str:
    if smtp:
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = smtp["sender"], email, subject
        msg.set_content(body)
        with smtplib.SMTP(smtp["host"], smtp["port"], timeout=30) as server:
            server.starttls()
            if smtp["user"]:
                server.login(smtp["user"], smtp["password"])
            server.send_message(msg)
        return "email"
    os.makedirs(os.path.dirname(outbox_path) or ".", exist_ok=True)
    with open(outbox_path, "a", encoding="utf-8") as fh:
        fh.write(f"--- {datetime.now(timezone.utc).isoformat(timespec='seconds')}\nTo: {email}\nSubject: {subject}\n\n{body}\n")
    return "outbox"


def check_alerts(alerts_db: str, catalog_db: str, outbox_path: str, verbose: bool = True) -> dict:
    stats = {"active": 0, "triggered": 0, "sent_by_email": 0, "written_to_outbox": 0, "failed": 0}
    if not os.path.exists(alerts_db) or not os.path.exists(catalog_db):
        return stats
    conn = sqlite3.connect(alerts_db, timeout=10)
    conn.row_factory = sqlite3.Row
    has_table = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='price_alerts'").fetchone()
    if not has_table:
        conn.close()
        return stats

    store = CatalogStore(catalog_db)
    smtp = _smtp_settings()
    alerts = conn.execute("SELECT * FROM price_alerts WHERE active=1").fetchall()
    stats["active"] = len(alerts)
    for alert in alerts:
        row = store.db.execute("SELECT data, active FROM products WHERE id=?", (alert["product_id"],)).fetchone()
        if row is None or not row["active"]:
            continue
        product = json.loads(row["data"])
        if not product.get("in_stock", True) or float(product["price"]) > float(alert["target_price"]):
            continue
        stats["triggered"] += 1
        subject, body = _message(alert, product)
        try:
            how = _deliver(alert["email"], subject, body, smtp, outbox_path)
        except Exception as exc:
            stats["failed"] += 1
            if verbose:
                print(f"  could not email {alert['email']}: {exc}")
            continue
        stats["sent_by_email" if how == "email" else "written_to_outbox"] += 1
        conn.execute("UPDATE price_alerts SET active=0, notified_at=?, notified_price=? WHERE id=?",
                     (datetime.now(timezone.utc).isoformat(timespec="seconds"), float(product["price"]), alert["id"]))
        conn.commit()
        if verbose:
            print(f"  alert {alert['id']}: {product.get('name')} is ₹{float(product['price']):,.0f} "
                  f"(target ₹{float(alert['target_price']):,.0f}) -> {'emailed' if how == 'email' else 'outbox'} {alert['email']}")
    store.close()
    conn.close()
    return stats
