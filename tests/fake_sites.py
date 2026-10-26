"""
Small local web servers that behave like real stores, for end-to-end crawler tests:
a Shopify store with a paginated /products.json, a store whose robots.txt blocks
crawlers, a site with sitemaps + JSON-LD product pages, a rate-limiting store,
and a store that refuses bots outright.
"""

from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

SHOPIFY_ROBOTS = "User-agent: *\nDisallow: /checkout\nDisallow: /collections/*sort_by*\nDisallow: /search\n"

GARMENTS = [
    ("Silk Saree", "Sarees", "Red"), ("Cotton Kurta", "Kurtas", "Blue"), ("Linen Shirt", "Shirts", "White"),
    ("Slim Fit Jeans", "Bottom", "Navy Blue"), ("Printed Dress", "Dresses", "Pink"),
    ("Leather Sneakers", "Shoes", "Black"), ("Oversized T-Shirt", "", "Olive"),
]


def make_image_bytes(color=(200, 30, 30)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, "JPEG")
    return buf.getvalue()


class FakeSite:
    """routes: path -> (status, content_type, body) or callable(handler, query) -> tuple."""

    def __init__(self):
        self.routes: dict = {}
        self.hits: list[str] = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.server.shutdown()

    def _handler(self):
        site = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                parts = urlsplit(self.path)
                site.hits.append(self.path)
                route = site.routes.get(parts.path)
                if route is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                if callable(route):
                    route = route(self, parse_qs(parts.query))
                status, ctype, body, *extra = route
                headers = extra[0] if extra else {}
                if isinstance(body, str):
                    body = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

        return Handler


class ShopifyStore(FakeSite):
    """A store with `count` products; prices can be changed between crawls."""

    def __init__(self, count: int, image_host: str | None = None):
        super().__init__()
        self.products = []
        for i in range(count):
            title, ptype, color = GARMENTS[i % len(GARMENTS)]
            self.products.append({
                "id": 1000 + i, "title": f"{color} {title} {i}", "handle": f"item-{i}", "vendor": "Fake Brand",
                "product_type": ptype, "tags": ["New Arrival", f"SKU{i}"], "body_html": f"<p>Soft {title.lower()}</p>",
                "options": [{"name": "Size", "values": ["S", "M"]}],
                "variants": [{"price": f"{999 + i}.00", "compare_at_price": f"{1499 + i}.00", "available": True,
                              "option1": "S"},
                             {"price": f"{999 + i}.00", "compare_at_price": f"{1499 + i}.00", "available": i % 5 != 0,
                              "option1": "M"}],
                "images": [{"position": 1, "src": f"{image_host}/img/{i}.jpg" if image_host else ""}],
            })
        self.routes["/robots.txt"] = (200, "text/plain", SHOPIFY_ROBOTS)
        self.routes["/products.json"] = self._feed

    def _feed(self, handler, query):
        limit = int(query.get("limit", ["30"])[0])
        page = int(query.get("page", ["1"])[0])
        chunk = self.products[(page - 1) * limit: page * limit]
        return 200, "application/json", json.dumps({"products": chunk})


def jsonld_page(name, price, sku, available=True):
    data = {"@context": "https://schema.org", "@type": "Product", "name": name, "sku": sku,
            "brand": {"@type": "Brand", "name": "Struct Brand"}, "image": f"/img/{sku}.jpg",
            "offers": {"@type": "Offer", "price": str(price), "priceCurrency": "INR",
                       "availability": "https://schema.org/" + ("InStock" if available else "OutOfStock")}}
    return f'<html><head><script type="application/ld+json">{json.dumps(data)}</script></head><body></body></html>'
