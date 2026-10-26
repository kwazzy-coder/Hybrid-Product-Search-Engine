"""
Run the real crawler CLI against local fake stores and check the whole flow:
robots.txt respected, pagination, rate limiting, refusals, sitemap + JSON-LD,
export, thumbnails, and price history across two crawls.
"""

import json
import os
import sqlite3

import pytest

from crawler.__main__ import main
from tests.fake_sites import FakeSite, ShopifyStore, jsonld_page, make_image_bytes


@pytest.fixture
def sites():
    started = []

    def start(site):
        started.append(site.start())
        return site

    images = FakeSite()
    images.routes["/robots.txt"] = (200, "text/plain", "User-agent: *\nAllow: /\n")
    jpeg = make_image_bytes()
    for i in range(300):
        images.routes[f"/img/{i}.jpg"] = (200, "image/jpeg", jpeg)
    start(images)

    big = start(ShopifyStore(260, image_host=images.url))          # needs 2 pages of 250

    closed = ShopifyStore(5)
    closed.routes["/robots.txt"] = (200, "text/plain", "User-agent: *\nDisallow: /\n")
    start(closed)

    struct = FakeSite()
    struct.routes["/robots.txt"] = (200, "text/plain", "User-agent: *\nDisallow: /cart\nSitemap: {url}/sitemap.xml\n")
    struct.routes["/p/kurta-1"] = (200, "text/html", jsonld_page("Women Yellow Cotton Kurta", 1299, "K1"))
    struct.routes["/p/dress-2"] = (200, "text/html", jsonld_page("Black Maxi Dress", 2499, "D2", available=False))
    struct.routes["/p/no-data"] = (200, "text/html", "<html><body>Nothing structured</body></html>")
    start(struct)
    sm_index = f"""<?xml version="1.0"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <sitemap><loc>{struct.url}/sitemap_products_1.xml</loc></sitemap></sitemapindex>"""
    sm_products = f"""<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>{struct.url}/p/kurta-1</loc></url><url><loc>{struct.url}/p/dress-2</loc></url>
        <url><loc>{struct.url}/p/no-data</loc></url><url><loc>{struct.url}/about</loc></url></urlset>"""
    struct.routes["/robots.txt"] = (200, "text/plain", f"User-agent: *\nDisallow: /cart\nSitemap: {struct.url}/sitemap.xml\n")
    struct.routes["/sitemap.xml"] = (200, "application/xml", sm_index)
    struct.routes["/sitemap_products_1.xml"] = (200, "application/xml", sm_products)

    slow = ShopifyStore(3)
    feed = slow.routes["/products.json"]
    calls = {"n": 0}

    def rate_limited(handler, query):
        calls["n"] += 1
        if calls["n"] == 1:
            return 429, "text/plain", "slow down", {"Retry-After": "1"}
        return feed(handler, query)

    slow.routes["/products.json"] = rate_limited
    start(slow)

    refuser = FakeSite()
    refuser.routes["/robots.txt"] = (200, "text/plain", "User-agent: *\nAllow: /\n")
    refuser.routes["/products.json"] = (403, "text/plain", "bots not welcome")
    start(refuser)

    yield {"images": images, "big": big, "closed": closed, "struct": struct, "slow": slow, "refuser": refuser}
    for s in started:
        s.stop()


def write_config(tmp_path, sites):
    cfg = f"""
[crawler]
user_agent = "HybridSearchBot/1.0"
contact = "test@example.com"
min_delay_seconds = 0.01
image_delay_seconds = 0.0
max_products_per_site = 1000
cache_dir = "{tmp_path}/cache"
db_path = "{tmp_path}/crawler.db"
export_path = "{tmp_path}/products_live.json"
images_dir = "{tmp_path}/images"

[[site]]
id = "big"
name = "Big Store"
url = "{sites['big'].url}"
platform = "shopify"

[[site]]
id = "closed"
name = "Closed Store"
url = "{sites['closed'].url}"
platform = "shopify"

[[site]]
id = "struct"
name = "Struct Store"
url = "{sites['struct'].url}"
platform = "sitemap"
product_url_pattern = "/p/"

[[site]]
id = "slow"
name = "Slow Store"
url = "{sites['slow'].url}"
platform = "auto"

[[site]]
id = "refuser"
name = "Refuser"
url = "{sites['refuser'].url}"
platform = "shopify"
"""
    path = tmp_path / "sites.toml"
    path.write_text(cfg)
    return str(path)


def test_crawl_export_images_and_price_history(tmp_path, sites, capsys):
    cfg = write_config(tmp_path, sites)
    assert main(["--config", cfg, "crawl"]) == 0

    db = sqlite3.connect(tmp_path / "crawler.db")
    runs = {site: (status, saved) for site, status, saved in
            db.execute("SELECT site, status, saved FROM crawl_runs")}
    assert runs["big"] == ("ok", 260)
    assert runs["closed"][0] == "disallowed"
    assert runs["struct"] == ("ok", 2)
    assert runs["slow"] == ("ok", 3)
    assert runs["refuser"][0] == "blocked"

    # robots.txt was honoured: the closed store's feed was never requested.
    assert not any(h.startswith("/products.json") for h in sites["closed"].hits)
    # Only URLs matching the product pattern were fetched from the sitemap site.
    assert "/about" not in sites["struct"].hits
    # Pagination: two pages of 250.
    assert sum(h.startswith("/products.json") for h in sites["big"].hits) == 2

    assert main(["--config", cfg, "export"]) == 0
    products = json.loads((tmp_path / "products_live.json").read_text())
    assert len(products) == 265
    by_id = {p["_id"]: p for p in products}
    kurta = next(p for p in products if p["source"] == "struct" and p["category"] == "kurta")
    assert kurta["brand"] == "Struct Brand" and kurta["price"] == 1299 and kurta["gender"] == "women"
    assert kurta["image_url"].startswith(sites["struct"].url)
    assert by_id["big_1000"]["category"] == "saree"
    assert by_id["big_1006"]["category"] == "tshirt"          # empty product_type, inferred from title

    assert main(["--config", cfg, "images", "--limit", "20"]) == 0
    assert len(os.listdir(tmp_path / "images")) == 20

    # Second crawl: one price drops, one product is taken down.
    sites["big"].products[0]["variants"][0]["price"] = "799.00"
    sites["big"].products[0]["variants"][1]["price"] = "799.00"
    del sites["big"].products[1]
    assert main(["--config", cfg, "crawl", "--site", "big", "--no-cache"]) == 0
    main(["--config", cfg, "export"])
    products = {p["_id"]: p for p in json.loads((tmp_path / "products_live.json").read_text())}
    assert "big_1001" not in products
    hist = products["big_1000"]["price_history"]
    assert products["big_1000"]["price"] == 799
    assert hist["previous_price"] == 999 and hist["price_dropped"] is True and hist["is_lowest_30d"] is True
    assert hist["points"] == 2

    out = capsys.readouterr().out
    assert "refused access (403)" in out
    assert "said 429" in out
