"""
Shopify stores publish their catalog at /products.json (and per collection at
/collections/<handle>/products.json), 250 products per page. This is the same
public feed the storefront itself uses, so it is far lighter on the site than
loading every product page.
"""

from __future__ import annotations

import json
from typing import Iterator

from ..http import PoliteSession

PAGE_SIZE = 250


class NotShopify(Exception):
    pass


def _raw_from_shopify(p: dict, base: str) -> dict:
    tags = p.get("tags") or []
    if isinstance(tags, str):  # older stores return a comma-separated string
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    images = sorted(p.get("images") or [], key=lambda img: img.get("position") or 0)
    return {
        "source_product_id": str(p.get("id")),
        "url": f"{base}/products/{p.get('handle')}",
        "title": p.get("title") or "",
        "vendor": p.get("vendor") or "",
        "product_type": p.get("product_type") or "",
        "tags": tags,
        "description_html": p.get("body_html") or "",
        "images": [img.get("src") for img in images if img.get("src")],
        "options": [{"name": o.get("name", ""), "values": o.get("values", [])} for o in p.get("options") or []],
        "variants": [
            {
                "price": v.get("price"),
                "compare_at_price": v.get("compare_at_price"),
                "available": v.get("available", True),
                "options": [v.get("option1"), v.get("option2"), v.get("option3")],
                "sku": v.get("sku"),
            }
            for v in p.get("variants") or []
        ],
        "updated_at": p.get("updated_at"),
    }


class ShopifyCrawl:
    """Iterate a store's products. `complete` is True if we reached the end of the catalog."""

    def __init__(self, session: PoliteSession, site, max_products: int):
        self.session = session
        self.site = site
        self.max_products = max_products
        self.complete = False
        self.note = ""
        self.base = site.url.rstrip("/")

    def __iter__(self) -> Iterator[dict]:
        feeds = [f"/collections/{c}/products.json" for c in self.site.collections] or ["/products.json"]
        seen: set[str] = set()
        reached_end_everywhere = True

        for feed in feeds:
            page = 1
            while True:
                if len(seen) >= self.max_products:
                    reached_end_everywhere = False
                    break
                # Page 1 without a page parameter: some stores' robots.txt blocks
                # "?page=" / "&page=" URLs for bots (e.g. "Disallow: /*&page=").
                url = f"{self.base}{feed}?limit={PAGE_SIZE}" + (f"&page={page}" if page > 1 else "")
                if page > 1 and not self.session.allowed(url):
                    self.note = (f"robots.txt doesn't allow paging past page {page - 1}, "
                                 f"so only the first {len(seen)} products were collected")
                    reached_end_everywhere = False
                    break
                res = self.session.get(url)
                if res.status == 404:
                    if page == 1 and feed == "/products.json":
                        raise NotShopify(f"{self.base} has no /products.json")
                    break
                if res.status != 200:
                    self.session.log(f"  {url} -> HTTP {res.status}; stopping this feed")
                    reached_end_everywhere = False
                    break
                try:
                    data = res.json()
                except json.JSONDecodeError:
                    raise NotShopify(f"{url} did not return JSON")
                products = data.get("products") if isinstance(data, dict) else None
                if products is None:
                    raise NotShopify(f"{url} is not a Shopify product feed")
                if not products:
                    break
                tag = " (cached)" if res.from_cache else ""
                self.session.log(f"  page {page}: {len(products)} products{tag}")
                for p in products:
                    pid = str(p.get("id"))
                    if pid in seen:
                        continue
                    seen.add(pid)
                    yield _raw_from_shopify(p, self.base)
                    if len(seen) >= self.max_products:
                        break
                if len(products) < PAGE_SIZE:
                    break
                page += 1

        self.complete = reached_end_everywhere


def probe(session: PoliteSession, base_url: str) -> dict | None:
    """Return the first product if base_url is a Shopify store, else None."""
    url = base_url.rstrip("/") + "/products.json?limit=1"
    res = session.get(url, use_cache=False)
    if res.status != 200:
        return None
    try:
        data = res.json()
    except json.JSONDecodeError:
        return None
    products = data.get("products") if isinstance(data, dict) else None
    if not products:
        return None
    return _raw_from_shopify(products[0], base_url.rstrip("/"))
