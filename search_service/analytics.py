"""Small in-process analytics store for the search demo.

For production this interface can be backed by Kafka/ClickHouse, but keeping the
contract here makes the metrics endpoint and event schema easy to demonstrate.
"""
from collections import Counter, deque
from statistics import mean
from threading import Lock
from time import time


class SearchAnalytics:
    def __init__(self, max_events: int = 5000):
        self._events = deque(maxlen=max_events)
        self._lock = Lock()

    def record_search(self, query_id: str, query: str, latency_ms: float, total: int):
        with self._lock:
            self._events.append({
                "type": "search", "query_id": query_id, "query": query,
                "latency_ms": round(latency_ms, 2), "total": total, "at": time(),
            })

    def record_click(self, query_id: str, product_id: str):
        with self._lock:
            self._events.append({
                "type": "click", "query_id": query_id, "product_id": product_id, "at": time(),
            })

    def summary(self):
        with self._lock:
            events = list(self._events)
        searches = [event for event in events if event["type"] == "search"]
        clicks = [event for event in events if event["type"] == "click"]
        search_ids = {event["query_id"] for event in searches}
        clicked_queries = {event["query_id"] for event in clicks}
        return {
            "searches": len(searches),
            "clicks": len(clicks),
            "click_through_rate": round(len(clicked_queries & search_ids) / len(search_ids), 4) if search_ids else 0.0,
            "zero_result_rate": round(sum(event["total"] == 0 for event in searches) / len(searches), 4) if searches else 0.0,
            "avg_latency_ms": round(mean(event["latency_ms"] for event in searches), 2) if searches else 0.0,
            "top_queries": [
                {"query": query, "count": count}
                for query, count in Counter(event["query"] for event in searches).most_common(5)
            ],
        }
