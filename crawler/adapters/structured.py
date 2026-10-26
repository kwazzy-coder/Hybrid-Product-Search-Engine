"""
Generic adapter for non-Shopify stores.

1. Find product pages through the site's sitemaps (listed in robots.txt or
   configured explicitly), keeping only URLs that match `product_url_pattern`.
2. On each page, read the schema.org Product data that stores embed for Google
   (<script type="application/ld+json">). If there is none, fall back to Open
   Graph product meta tags.

Because this reads standard markup rather than CSS selectors, one parser works
across many sites and doesn't break when a site redesigns.
"""

from __future__ import annotations

import gzip
import json
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from typing import Iterator
from urllib.parse import urljoin

from ..http import PoliteSession, RobotsDisallowed


# ------------------------------------------------------------- HTML parsing
class _PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.jsonld: list[str] = []
        self.meta: dict[str, str] = {}
        self._in_jsonld = False
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "script" and "ld+json" in a.get("type", "").lower():
            self._in_jsonld, self._buf = True, []
        elif tag == "meta":
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
            if key and "content" in a and key not in self.meta:
                self.meta[key] = a["content"]

    def handle_endtag(self, tag):
        if tag == "script" and self._in_jsonld:
            self.jsonld.append("".join(self._buf))
            self._in_jsonld = False

    def handle_data(self, data):
        if self._in_jsonld:
            self._buf.append(data)


def _loads_lenient(text: str):
    text = text.strip()
    for candidate in (text, re.sub(r",\s*([}\]])", r"\1", text)):
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            continue
    return None


def _walk(node, depth: int = 0):
    if depth > 6:
        return
    if isinstance(node, list):
        for item in node:
            yield from _walk(item, depth + 1)
    elif isinstance(node, dict):
        yield node
        for value in node.values():
            if isinstance(value, (dict, list)):
                yield from _walk(value, depth + 1)


def _is_type(node: dict, name: str) -> bool:
    types = node.get("@type")
    types = types if isinstance(types, list) else [types]
    return any(isinstance(t, str) and t.split("/")[-1].split(":")[-1] == name for t in types)


def _text(value) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or value.get("@value") or "")
    if isinstance(value, list):
        return _text(value[0]) if value else ""
    return str(value or "")


def _images(value) -> list[str]:
    out = []
    for item in value if isinstance(value, list) else [value]:
        if isinstance(item, dict):
            item = item.get("url") or item.get("contentUrl")
        if isinstance(item, str) and item:
            out.append(item)
    return out


def _availability(value) -> bool:
    text = _text(value).lower()
    if not text:
        return True
    return not any(word in text for word in ("outofstock", "soldout", "discontinued", "out of stock"))


def _offers(node: dict) -> list[dict]:
    offers = node.get("offers")
    flat: list[dict] = []
    for offer in offers if isinstance(offers, list) else [offers]:
        if not isinstance(offer, dict):
            continue
        if _is_type(offer, "AggregateOffer") and offer.get("offers"):
            flat.extend(o for o in offer["offers"] if isinstance(o, dict))
        else:
            flat.append(offer)

    variants = []
    for offer in flat:
        spec = offer.get("priceSpecification")
        specs = spec if isinstance(spec, list) else [spec] if isinstance(spec, dict) else []
        price = offer.get("price") or offer.get("lowPrice")
        list_price = None
        for s in specs:
            kind = _text(s.get("priceType")).lower()
            if "strikethrough" in kind or "listprice" in kind:
                list_price = s.get("price")
            elif price is None:
                price = s.get("price")
        variants.append({
            "price": price,
            "compare_at_price": list_price,
            "available": _availability(offer.get("availability")),
            "options": [_text(offer.get("size")) or None],
            "sku": offer.get("sku"),
            "currency": offer.get("priceCurrency") or (specs[0].get("priceCurrency") if specs else None),
        })
    return variants


def extract_product(page_html: str, url: str) -> dict | None:
    """Pull one product out of a product page. Returns a raw product or None."""
    parser = _PageParser()
    try:
        parser.feed(page_html)
    except Exception:
        return None

    product_node = None
    for block in parser.jsonld:
        data = _loads_lenient(block)
        if data is None:
            continue
        for node in _walk(data):
            if _is_type(node, "Product") or _is_type(node, "ProductGroup"):
                product_node = node
                break
        if product_node:
            break

    if product_node:
        variants = _offers(product_node)
        if not variants and isinstance(product_node.get("hasVariant"), list):
            for v in product_node["hasVariant"]:
                if isinstance(v, dict):
                    for off in _offers(v):
                        off["options"] = [_text(v.get("size")) or None]
                        variants.append(off)
        rating = product_node.get("aggregateRating") or {}
        sizes = sorted({o["options"][0] for o in variants if o["options"][0]})
        return {
            "source_product_id": _text(product_node.get("sku") or product_node.get("productID")
                                       or product_node.get("productGroupID")) or url,
            "url": urljoin(url, _text(product_node.get("url")) or parser.meta.get("og:url") or url),
            "title": _text(product_node.get("name")) or parser.meta.get("og:title", ""),
            "brand": _text(product_node.get("brand")),
            "product_type": _text(product_node.get("category")),
            "tags": [],
            "description_html": _text(product_node.get("description")) or parser.meta.get("og:description", ""),
            "images": [urljoin(url, i) for i in (_images(product_node.get("image")) or _images(parser.meta.get("og:image")))],
            "options": [{"name": "Size", "values": sizes}] if sizes else [],
            "variants": variants,
            "color": _text(product_node.get("color")),
            "rating": float(rating.get("ratingValue") or 0) if isinstance(rating, dict) else 0.0,
            "review_count": int(float(rating.get("reviewCount") or rating.get("ratingCount") or 0)) if isinstance(rating, dict) else 0,
            "currency": next((v["currency"] for v in variants if v.get("currency")), None),
            "gtin": _text(product_node.get("gtin13") or product_node.get("gtin") or product_node.get("gtin12")),
        }

    # Fallback: Open Graph product tags.
    meta = parser.meta
    price = meta.get("product:price:amount") or meta.get("og:price:amount") or meta.get("product:sale_price:amount")
    if not price or not meta.get("og:title"):
        return None
    availability = meta.get("product:availability") or meta.get("og:availability") or ""
    return {
        "source_product_id": meta.get("product:retailer_item_id") or url,
        "url": urljoin(url, meta.get("og:url") or url),
        "title": meta.get("og:title", ""),
        "brand": meta.get("product:brand", ""),
        "product_type": meta.get("product:category", ""),
        "tags": [],
        "description_html": meta.get("og:description", ""),
        "images": [urljoin(url, i) for i in _images(meta.get("og:image"))],
        "options": [],
        "variants": [{
            "price": price,
            "compare_at_price": meta.get("product:original_price:amount"),
            "available": _availability(availability),
            "options": [None],
        }],
        "color": meta.get("product:color", ""),
        "currency": meta.get("product:price:currency") or meta.get("og:price:currency"),
    }


# ------------------------------------------------------------------ sitemaps
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_sitemap(xml_bytes: bytes) -> tuple[list[str], list[str]]:
    """Return (child sitemap URLs, page URLs)."""
    if xml_bytes[:2] == b"\x1f\x8b":
        xml_bytes = gzip.decompress(xml_bytes)
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return [], []
    children, pages = [], []
    kind = _local(root.tag)
    for el in root:
        loc = next((c.text.strip() for c in el if _local(c.tag) == "loc" and c.text), None)
        if not loc:
            continue
        (children if kind == "sitemapindex" else pages).append(loc)
    return children, pages


class SitemapCrawl:
    def __init__(self, session: PoliteSession, site, max_products: int):
        self.session = session
        self.site = site
        self.max_products = max_products
        self.complete = False
        self.base = site.url.rstrip("/")
        self.pattern = re.compile(site.product_url_pattern) if site.product_url_pattern else None

    def _product_urls(self) -> Iterator[str]:
        queue = list(self.site.sitemaps) or list(self.session.robots_for(self.base).sitemaps) or [self.base + "/sitemap.xml"]
        visited: set[str] = set()
        while queue:
            sm = queue.pop(0)
            if sm in visited:
                continue
            visited.add(sm)
            try:
                body = self.session.get_bytes(sm) if sm.endswith(".gz") else self.session.get(sm).text.encode("utf-8")
            except RobotsDisallowed:
                continue
            if not body:
                continue
            children, pages = parse_sitemap(body)
            # Visit product sitemaps first when the index names them.
            children.sort(key=lambda u: 0 if "product" in u.lower() else 1)
            queue.extend(children)
            for page in pages:
                if self.pattern is None or self.pattern.search(page):
                    yield page

    def __iter__(self) -> Iterator[dict]:
        count = 0
        seen: set[str] = set()
        exhausted = True
        for url in self._product_urls():
            if count >= self.max_products:
                exhausted = False
                break
            if url in seen:
                continue
            seen.add(url)
            try:
                res = self.session.get(url)
            except RobotsDisallowed:
                continue
            if res.status != 200:
                continue
            raw = extract_product(res.text, url)
            if raw is None:
                continue
            count += 1
            if count % 25 == 0:
                self.session.log(f"  {count} product pages read")
            yield raw
        self.complete = exhausted
