"""
Build Hard Negatives Dataset from Click Events
==============================================
Reads click_events from SQLite database (feedback.db) and constructs pairwise/listwise
training tuples for Learning-to-Rank (LTR).

Logic:
- For each search session:
  - Clicked item (clicked_id) -> Positive label (3)
  - Any item ranked in top 10 of shown_results that was NOT clicked -> Hard negative label (0)

Output:
- Saves processed dataset to data/hard_negatives.json
"""

import os
import sys
import json
from collections import defaultdict

# Add parent directory to path so we can import feedback_db
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feedback_db import get_all_click_events, init_db

def build_hard_negatives():
    print("=" * 60)
    print("  Building Hard Negatives Dataset from SQLite Click Events")
    print("=" * 60)

    init_db()
    click_events = get_all_click_events()
    print(f"Retrieved {len(click_events)} click events from SQLite.")

    if not click_events:
        print("\n[WARNING] No click events found in click_events table.")
        print("Run scripts/simulate_clicks.py first to generate click session data.")
        return []

    processed_sessions = []
    total_positives = 0
    total_negatives = 0

    for idx, event in enumerate(click_events):
        query = event["query"]
        shown_results = event.get("shown_results", [])
        clicked_id = event.get("clicked_id")

        if not query or not clicked_id or not shown_results:
            continue

        # Top 10 items evaluated for relevance
        top_10_shown = shown_results[:10]

        items = []
        # Positive label for clicked item
        items.append({
            "product_id": clicked_id,
            "label": 3,
            "is_clicked": True
        })
        total_positives += 1

        # Hard negatives for unclicked items in top 10
        for rank, pid in enumerate(top_10_shown):
            if pid != clicked_id:
                items.append({
                    "product_id": pid,
                    "label": 0,
                    "is_clicked": False,
                    "shown_rank": rank + 1
                })
                total_negatives += 1

        processed_sessions.append({
            "session_id": event.get("id", idx + 1),
            "query": query,
            "clicked_id": clicked_id,
            "timestamp": event.get("timestamp"),
            "items": items
        })

    # Save to data/hard_negatives.json
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_data_dir = os.path.join(os.path.dirname(current_dir), "data")
    os.makedirs(parent_data_dir, exist_ok=True)
    output_path = os.path.join(parent_data_dir, "hard_negatives.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_sessions, f, indent=2)

    print("\n" + "=" * 60)
    print("  Hard Negatives Dataset Successfully Built!")
    print("=" * 60)
    print(f"  Sessions Processed:     {len(processed_sessions)}")
    print(f"  Positive Click Labels:  {total_positives}")
    print(f"  Hard Negative Labels:   {total_negatives}")
    print(f"  Total Labelled Pairs:   {total_positives + total_negatives}")
    print(f"  Dataset Saved To:       {output_path}\n")

    return processed_sessions

if __name__ == "__main__":
    build_hard_negatives()
