"""
Simulate Synthetic Click Sessions CLI
======================================
Simulates ~50-100 search + click sessions across the catalog to populate
the click_events SQLite table (feedback.db) for LTR training.

Usage:
    python scripts/simulate_clicks.py [--sessions 75]
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime, timedelta

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_understanding import parse_query_local_fallback
from feedback_db import log_click_event, init_db

# Representative query bank
QUERY_BANK = [
    "red dress", "blue shirt", "black shoes", "white tshirt", "green kurta",
    "formal shoes", "casual jeans", "sports shoes", "pink dress", "navy blue shirt",
    "running shoes", "denim jeans", "cotton shirt", "summer dress", "winter jacket",
    "formal shirt", "party dress", "leather shoes", "sneakers", "sandals",
    "ethnic kurta", "casual wear", "office wear", "blue jeans", "black formal shoes",
    "red top", "floral dress", "striped shirt", "polo tshirt", "track pants"
]

def load_products():
    """Load product catalog."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "data", "products_2k.json")
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(current_dir), "data", "products_2k.json")
    
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)

def rank_products_for_query(query: str, products: list) -> list:
    """Simple ranker matching query intent to retrieve top candidates for simulation."""
    intent = parse_query_local_fallback(query)
    q_cat = (intent.category or "").lower()
    q_colors = [c.lower() for c in (intent.color or [])]
    q_terms = [t.lower() for t in intent.expanded_terms]
    
    scored_products = []
    for p in products:
        p_name = str(p.get("name", "")).lower()
        p_cat = str(p.get("category", "")).lower()
        p_subcat = str(p.get("subcategory", "")).lower()
        
        p_color_val = p.get("color", [])
        if not isinstance(p_color_val, list):
            p_color_val = [p_color_val]
        p_colors = [str(c).lower() for c in p_color_val if c]
        
        score = 0.0
        # Category match
        if q_cat and (q_cat == p_cat or q_cat in p_subcat or q_cat in p_name):
            score += 3.0
        # Color match
        if any(c in p_colors for c in q_colors) or any(c in p_name for c in q_colors):
            score += 2.5
        # Term match in name
        for term in q_terms:
            if term in p_name:
                score += 1.0
        # Rating boost
        score += float(p.get("rating", 0)) * 0.2
        # Random noise to simulate retrieval variability
        score += random.uniform(0.0, 0.5)
        
        scored_products.append((p["_id"], score))
        
    scored_products.sort(key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in scored_products[:15]]

def simulate_clicks(num_sessions: int = 75):
    print("=" * 60)
    print(f"  Simulating {num_sessions} Search & Click Sessions")
    print("=" * 60)
    
    init_db()
    products = load_products()
    product_map = {p["_id"]: p for p in products}
    print(f"Loaded catalog with {len(products)} products.")
    
    random.seed(42)
    start_time = datetime.now() - timedelta(days=2)
    logged_count = 0
    
    for i in range(num_sessions):
        query = random.choice(QUERY_BANK)
        candidate_ids = rank_products_for_query(query, products)
        
        if not candidate_ids:
            continue
            
        top_shown = candidate_ids[:10]
        
        # Simulate user click selection (biased towards items matching intent & higher rank)
        intent = parse_query_local_fallback(query)
        q_cat = (intent.category or "").lower()
        q_colors = [c.lower() for c in (intent.color or [])]
        
        weights = []
        for rank, pid in enumerate(top_shown):
            p = product_map.get(pid, {})
            p_name = str(p.get("name", "")).lower()
            
            p_color_val = p.get("color", [])
            if not isinstance(p_color_val, list):
                p_color_val = [p_color_val]
            p_colors = [str(c).lower() for c in p_color_val if c]
            
            # Position decay bias
            w = 1.0 / (rank + 1)
            
            # Relevance boost
            if q_cat and (q_cat in p_name or q_cat == str(p.get("category","")).lower()):
                w *= 2.0
            if any(c in p_colors for c in q_colors) or any(c in p_name for c in q_colors):
                w *= 2.0
                
            weights.append(w)
            
        clicked_id = random.choices(top_shown, weights=weights, k=1)[0]
        
        # Simulated timestamp spread over 2 days
        simulated_ts = (start_time + timedelta(minutes=i * 35)).strftime("%Y-%m-%d %H:%M:%S")
        
        event_id = log_click_event(
            query=query,
            shown_results=top_shown,
            clicked_id=clicked_id,
            timestamp=simulated_ts
        )
        logged_count += 1
        
        if (i + 1) % 15 == 0 or i + 1 == num_sessions:
            print(f"  Simulated {i + 1}/{num_sessions} sessions... (Last Query: '{query}' -> Clicked ID: {clicked_id})")
            
    print("\n" + "=" * 60)
    print("  Click Session Simulation Complete!")
    print("=" * 60)
    print(f"  Total Click Sessions Logged: {logged_count}")
    print(f"  Destination Table:           click_events (SQLite DB: feedback.db)\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate synthetic click sessions")
    parser.add_argument("--sessions", type=int, default=75, help="Number of click sessions to simulate (default: 75)")
    args = parser.parse_args()
    
    simulate_clicks(num_sessions=args.sessions)
