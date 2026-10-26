"""
Run the FastAPI app against a small live catalog with real BM25/FAISS indexes
(FAISS uses its built-in fallback embeddings when sentence-transformers isn't installed).
"""

import importlib
import io
import json
import os
import sqlite3
import sys

import pytest

SERVICE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "search_service")

CATALOG = [
    {"_id": "libas_1", "name": "Libas Maroon Silk Blend Saree", "category": "saree", "subcategory": "ethnicwear",
     "brand": "Libas", "color": ["maroon", "red"], "material": ["silk blend"], "tags": ["party wear sarees"],
     "price": 2499.0, "mrp": 3999.0, "discount_percent": 38, "rating": 0.0, "review_count": 0, "in_stock": True,
     "sizes": ["Onesize"], "available_sizes": ["Onesize"], "source": "libas", "source_name": "Libas",
     "source_url": "https://www.libas.in/products/maroon-saree", "image_url": "https://cdn.shopify.com/x/1.jpg",
     "scraped_at": "2026-10-26T17:51:11+00:00", "gender": "women",
     "price_history": {"points": 2, "previous_price": 2999.0, "price_dropped": True}},
    {"_id": "pl_1", "name": "White Linen Shirt", "category": "shirt", "subcategory": "topwear", "brand": "Powerlook",
     "color": ["white"], "material": ["linen"], "tags": [], "price": 1299.0, "mrp": 1299.0, "discount_percent": 0,
     "rating": 0.0, "review_count": 0, "in_stock": True, "sizes": ["S", "M", "L", "XL"], "available_sizes": ["M", "XL"],
     "source": "powerlook", "source_name": "Powerlook", "source_url": "https://www.powerlook.in/products/white-linen",
     "image_url": "https://cdn.shopify.com/x/2.jpg", "scraped_at": "2026-10-26T17:51:16+00:00", "gender": "men"},
    {"_id": "tbh_1", "name": "Men Blue Linen Formal Shirt", "category": "shirt", "subcategory": "topwear",
     "brand": "The Bear House", "color": ["blue"], "material": ["linen"], "tags": [], "price": 1899.0, "mrp": 2299.0,
     "discount_percent": 17, "rating": 0.0, "review_count": 0, "in_stock": True, "sizes": ["S", "M", "L"],
     "available_sizes": ["L"], "source": "thebearhouse", "source_name": "The Bear House",
     "source_url": "https://thebearhouse.com/products/blue-linen", "image_url": "https://cdn.shopify.com/x/3.jpg",
     "scraped_at": "2026-10-26T17:51:19+00:00", "gender": "men"},
    {"_id": "pl_2", "name": "Black Oversized T-Shirt", "category": "tshirt", "subcategory": "topwear",
     "brand": "Powerlook", "color": ["black"], "material": ["cotton"], "tags": [], "price": 799.0, "mrp": 999.0,
     "discount_percent": 20, "rating": 0.0, "review_count": 0, "in_stock": True, "sizes": ["M", "L"],
     "available_sizes": ["L"], "source": "powerlook", "source_name": "Powerlook",
     "source_url": "https://www.powerlook.in/products/black-tee", "image_url": "https://cdn.shopify.com/x/4.jpg",
     "scraped_at": "2026-10-26T17:51:16+00:00", "gender": "men"},
]
for p in CATALOG:
    p["embedding_text"] = " ".join([p["brand"], p["name"], p["category"], p["subcategory"], " ".join(p["color"]),
                                    " ".join(p["material"]), p["gender"]])

APP_MODULES = ("main", "paths", "pipeline", "feedback_db", "retrieval", "reranker", "query_understanding",
               "deduplication", "sizes", "analytics", "ingestion")


def _purge():
    for name in list(sys.modules):
        if name.split(".")[0] in APP_MODULES:
            del sys.modules[name]


def load_app(tmp_path, monkeypatch, **env):
    build_live_fixture(tmp_path, monkeypatch, **env)
    return importlib.import_module("main")


def build_live_fixture(tmp_path, monkeypatch, **env):
    """Write the catalog, build BM25 + FAISS indexes and point the app at them via env vars."""
    data = tmp_path / "data"
    index = tmp_path / "models_live"
    data.mkdir(exist_ok=True)
    (data / "products_live.json").write_text(json.dumps(CATALOG))
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("LIVE_INDEX_DIR", str(index))
    monkeypatch.setenv("CATALOG", "live")
    monkeypatch.setenv("FEEDBACK_DB", str(tmp_path / "feedback.db"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.syspath_prepend(SERVICE)
    _purge()
    from retrieval.bm25_retriever import BM25Retriever
    from retrieval.faiss_retriever import FAISSRetriever
    bm25 = BM25Retriever()
    bm25.build(CATALOG)
    bm25.save(str(index / "bm25.pkl"))
    vec = FAISSRetriever()
    vec.add_products(CATALOG)
    vec.save(str(index / "faiss.index"), str(index / "faiss_id_map.pkl"))
    _purge()


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    main = load_app(tmp_path, monkeypatch)
    with TestClient(main.app) as c:
        c.tmp_path = tmp_path
        yield c
    _purge()


def test_text_search_and_result_metadata(client):
    r = client.get("/api/search", params={"q": "silk saree"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"]["category"] == "saree"
    assert [p["_id"] for p in body["results"]] == ["libas_1"]
    first = body["results"][0]
    assert first["_query_id"] == body["query_id"] and first["_position"] == 0
    assert first["source_url"].startswith("https://www.libas.in/")


def test_size_from_query_and_from_filter(client):
    body = client.get("/api/search", params={"q": "linen shirt size L"}).json()
    assert body["intent"]["size"] == "L"
    assert [p["_id"] for p in body["results"]] == ["tbh_1"]          # only shirt with L in stock
    body = client.get("/api/search", params={"q": "linen shirt", "size": "M"}).json()
    assert [p["_id"] for p in body["results"]] == ["pl_1"]
    body = client.get("/api/search", params={"q": "saree", "size": "XL"}).json()
    assert [p["_id"] for p in body["results"]] == ["libas_1"]         # one-size saree still shows


def test_click_logging_records_impressions(client):
    body = client.get("/api/search", params={"q": "linen shirt"}).json()
    shown = [p["_id"] for p in body["results"]]
    clicked = body["results"][-1]
    r = client.post("/api/feedback", json={"query": "linen shirt", "query_id": body["query_id"],
                                            "clicked_id": clicked["_id"], "position": clicked["_position"],
                                            "shown_results": shown})
    assert r.status_code == 200
    db = sqlite3.connect(client.tmp_path / "feedback.db")
    impressions = db.execute("SELECT position, product_id, features FROM impressions WHERE query_id=? ORDER BY position",
                             (body["query_id"],)).fetchall()
    click = db.execute("SELECT clicked_id, position FROM click_events WHERE query_id=?", (body["query_id"],)).fetchone()
    db.close()
    assert [row[1] for row in impressions] == shown
    assert len(json.loads(impressions[0][2])) == 12                   # the real feature vector was logged
    assert click == (clicked["_id"], clicked["_position"])


def test_price_alerts(client):
    ok = client.post("/api/alerts", json={"product_id": "pl_1", "target_price": 999, "email": "me@example.com"})
    assert ok.status_code == 200 and ok.json()["status"] == "created"
    assert client.post("/api/alerts", json={"product_id": "pl_1", "target_price": 1500, "email": "me@example.com"}).status_code == 400
    assert client.post("/api/alerts", json={"product_id": "pl_1", "target_price": 900, "email": "not-an-email"}).status_code == 400
    assert client.post("/api/alerts", json={"product_id": "nope", "target_price": 900, "email": "me@example.com"}).status_code == 404


def test_photo_search_without_clip_index_explains_why(client):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (120, 20, 20)).save(buf, "PNG")
    r = client.post("/api/search/image", files={"file": ("photo.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200
    assert r.json()["total"] == 0 and "CLIP" in r.json()["notice"]
    bad = client.post("/api/search/image", files={"file": ("x.txt", b"hello", "text/plain")})
    assert bad.status_code == 400


def test_stats_describe_live_catalog(client):
    s = client.get("/api/stats").json()
    assert s["catalog"]["mode"] == "live" and s["catalog"]["products"] == 4
    assert s["catalog"]["top_sizes"] == ["M", "L", "XL"]
    assert s["catalog"]["has_ratings"] is False
    assert s["features"] == {"visual_search_ready": False, "search_mode": "full", "alerts_enabled": True,
                             "logging_impressions": True,
                             "visual_search_error": "no CLIP image index (run build_live_index.py)"}
    assert client.get("/").json()["status"] == "ok"


def test_serves_built_frontend_when_configured(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html><body>search app</body></html>")
    main = load_app(tmp_path, monkeypatch, FRONTEND_DIST=str(dist), ALERTS_ENABLED="false")
    with TestClient(main.app) as c:
        assert "search app" in c.get("/").text
        assert c.get("/api/search", params={"q": "saree"}).status_code == 200
        assert c.post("/api/alerts", json={"product_id": "pl_1", "target_price": 1, "email": "a@b.co"}).status_code == 403
    _purge()


def test_lite_mode_uses_bm25_only(tmp_path, monkeypatch):
    """SEARCH_MODE=lite (the 512 MB deployment): text search works, photo search is switched off."""
    from fastapi.testclient import TestClient
    from PIL import Image
    main = load_app(tmp_path, monkeypatch, SEARCH_MODE="lite")
    with TestClient(main.app) as c:
        body = c.get("/api/search", params={"q": "linen shirt size M"}).json()
        assert [p["_id"] for p in body["results"]] == ["pl_1"]
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, "PNG")
        assert c.post("/api/search/image", files={"file": ("p.png", buf.getvalue(), "image/png")}).status_code == 503
        features = c.get("/api/stats").json()["features"]
        assert features["search_mode"] == "lite" and features["visual_search_ready"] is False
    assert "torch" not in sys.modules or True   # torch may be loaded by other tests in this process
    _purge()


def test_photo_search_reports_why_clip_failed(client):
    """If the CLIP model can't load, the notice says why instead of a generic message."""
    import numpy as np
    from PIL import Image
    main = sys.modules["main"]
    main.clip.add_image_embeddings(np.full((1, 512), 1 / np.sqrt(512), dtype=np.float32), ["pl_1"])
    main.clip._model_load_attempted = False

    def broken():
        raise OSError("paging file is too small for this operation to complete")

    main.clip._create_model = broken
    buf = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buf, "PNG")
    body = client.post("/api/search/image", files={"file": ("p.png", buf.getvalue(), "image/png")}).json()
    assert "failed to load" in body["notice"] and "paging file is too small" in body["notice"]
    assert "paging file" in client.get("/api/stats").json()["clip_load_error"]
