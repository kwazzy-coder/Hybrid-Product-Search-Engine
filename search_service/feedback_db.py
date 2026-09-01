import os
import json
import sqlite3
from typing import List, Optional, Dict, Any

def get_db_path() -> str:
    """Get absolute path to SQLite database file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_data_dir = os.path.join(os.path.dirname(current_dir), "data")
    if os.path.exists(parent_data_dir):
        return os.path.join(parent_data_dir, "feedback.db")
    
    local_data_dir = os.path.join(current_dir, "data")
    os.makedirs(local_data_dir, exist_ok=True)
    return os.path.join(local_data_dir, "feedback.db")

def init_db():
    """Initialize SQLite database and click_events table if not present."""
    db_path = get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS click_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            shown_results TEXT NOT NULL,
            clicked_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def log_click_event(query: str, shown_results: List[str], clicked_id: str, timestamp: Optional[str] = None) -> int:
    """Insert a new click event into the SQLite table."""
    init_db()
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    shown_json = json.dumps(shown_results)
    
    if timestamp:
        cursor.execute(
            "INSERT INTO click_events (query, shown_results, clicked_id, timestamp) VALUES (?, ?, ?, ?)",
            (query, shown_json, clicked_id, timestamp)
        )
    else:
        cursor.execute(
            "INSERT INTO click_events (query, shown_results, clicked_id) VALUES (?, ?, ?)",
            (query, shown_json, clicked_id)
        )
    conn.commit()
    event_id = cursor.lastrowid
    conn.close()
    return event_id

def get_all_click_events() -> List[Dict[str, Any]]:
    """Retrieve all logged click events from SQLite."""
    init_db()
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, query, shown_results, clicked_id, timestamp FROM click_events")
    rows = cursor.fetchall()
    conn.close()
    
    events = []
    for r in rows:
        events.append({
            "id": r[0],
            "query": r[1],
            "shown_results": json.loads(r[2]),
            "clicked_id": r[3],
            "timestamp": r[4]
        })
    return events
