"""
Turn a raw product (from any adapter) into the canonical schema the search
service indexes. The canonical fields match data/products_2k.json, plus a few
extras that only real data can provide (sizes, material, currency, images, ...).

Raw product shape produced by the adapters:
    {
      "source_product_id": str, "url": str, "title": str, "vendor": str,
      "product_type": str, "tags": [str], "description_html": str,
      "images": [str], "options": [{"name": str, "values": [str]}],
      "variants": [{"price": str|float, "compare_at_price": str|float|None,
                    "available": bool, "options": [str|None], "sku": str|None}],
      # optional, from structured data:
      "brand": str, "color": str, "rating": float, "review_count": int, "currency": str
    }
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone

from .taxonomy import (
    CATEGORY_GROUP, COLOR_PATTERN, COLORS, COMPILED_CATEGORIES, DOMINANT,
    GENDER_PATTERNS, MATERIAL_PATTERN, NON_FASHION,
)

TAG_NOISE = re.compile(r"[\d:_@/\\=]|^.{0,2}$")
WS = re.compile(r"\s+")
HTML_TAG = re.compile(r"<[^>]+>")
GENERIC_TYPES = {
    "top", "tops", "topwear", "top wear", "bottom", "bottoms", "bottomwear", "bottom wear", "apparel",
    "clothing", "clothes", "fashion", "men", "women", "unisex", "western wear", "ethnic wear", "new arrivals",
    "default", "product", "products", "all",
}


# ------------------------------------------------------------------ helpers
def to_price(value) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^\d.]", "", str(value).replace(",", ""))
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def strip_html(text: str | None, limit: int = 500) -> str:
    if not text:
        return ""
    text = html.unescape(HTML_TAG.sub(" ", text))
    text = WS.sub(" ", text).strip()
    return text[:limit].rsplit(" ", 1)[0] + "…" if len(text) > limit else text


def nice_case(text: str) -> str:
    """'RARE RABBIT' -> 'Rare Rabbit', 'powerlook-apparels' -> 'Powerlook Apparels'."""
    text = text.replace("-", " ").strip()
    if text.isupper() or text.islower():
        return " ".join(w.capitalize() for w in text.split())
    return text


def looks_like_code(text: str) -> bool:
    """'BEARSKY-RD', 'RR303158' — a SKU sitting where the title should be."""
    text = text.strip()
    return bool(text) and " " not in text and bool(re.fullmatch(r"[A-Z0-9][A-Z0-9\-_.]{2,}", text))


def clean_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    for tag in tags:
        tag = WS.sub(" ", str(tag)).strip()
        if not tag or TAG_NOISE.search(tag) or len(tag) > 30 or len(tag.split()) > 4:
            continue
        low = tag.lower()
        if low not in out:
            out.append(low)
    return out


# ---------------------------------------------------------------- inference
def infer_category(title: str, product_type: str, tags: list[str]) -> tuple[str | None, str | None]:
    """Score categories across fields; the garment noun at the end of a title is the strongest hint."""
    scores: dict[str, float] = {}

    # A specific product_type ("Sarees", "SHIRT") is the merchant's own label and
    # the strongest signal; a generic one ("Top", "Bottom") only breaks ties.
    type_weight = 0.5 if product_type.strip().lower() in GENERIC_TYPES else 3.0
    for name, _group, pattern in COMPILED_CATEGORIES:
        if product_type and pattern.search(product_type):
            scores[name] = scores.get(name, 0) + type_weight

    # Title: prefer the match that ends last (the head noun), then the longest.
    title_hits = []
    for order, (name, _group, pattern) in enumerate(COMPILED_CATEGORIES):
        for m in pattern.finditer(title or ""):
            title_hits.append((m.end(), m.end() - m.start(), -order, name))
    if title_hits:
        dominant = [h for h in title_hits if h[3] in DOMINANT]
        best = max(dominant or title_hits)
        scores[best[3]] = scores.get(best[3], 0) + (4.0 if dominant else 2.5)
        for hit in title_hits:
            if hit[3] != best[3]:
                scores[hit[3]] = scores.get(hit[3], 0) + 0.5

    tag_text = " | ".join(tags)
    for name, _group, pattern in COMPILED_CATEGORIES:
        if pattern.search(tag_text):
            scores[name] = scores.get(name, 0) + 1.0

    if not scores:
        return None, None
    order = {name: i for i, (name, _, _) in enumerate(COMPILED_CATEGORIES)}
    category = max(scores, key=lambda n: (scores[n], -order[n]))
    return category, CATEGORY_GROUP[category]


def infer_colors(title: str, tags: list[str], option_values: list[str], explicit: str | None) -> list[str]:
    found: list[str] = []

    def add(name: str) -> None:
        name = name.lower().replace("-", " ").strip()
        if name in COLORS:
            for value in (name, *COLORS[name]):
                if value not in found:
                    found.append(value)

    sources = [explicit or ""] + option_values + [title]
    # Tags like "Colour_White", "Color: Red", "Red", "Beige Sarees"
    for tag in tags:
        sources.append(re.sub(r"(?i)colou?r[_:\s-]*", " ", tag))
    for text in sources:
        for m in COLOR_PATTERN.finditer(text or ""):
            add(m.group(1))
    return found


def infer_gender(texts: list[str], default: str | None) -> str | None:
    joined = " | ".join(t for t in texts if t)
    hits = {g for g, pattern in GENDER_PATTERNS if pattern.search(joined)}
    if "unisex" in hits or {"men", "women"} <= hits:
        return "unisex"
    if hits:
        return hits.pop()
    return default


def infer_materials(*texts: str) -> list[str]:
    found: list[str] = []
    for text in texts:
        for m in MATERIAL_PATTERN.finditer(text or ""):
            value = m.group(1).lower()
            if value not in found:
                found.append(value)
    return found


def choose_price(variants: list[dict]) -> tuple[float | None, float | None, bool]:
    priced = [(to_price(v.get("price")), to_price(v.get("compare_at_price")), bool(v.get("available", True))) for v in variants]
    priced = [p for p in priced if p[0] is not None and p[0] > 0]
    if not priced:
        return None, None, False
    in_stock = any(avail for _, _, avail in priced)
    pool = [p for p in priced if p[2]] or priced
    price, compare_at, _ = min(pool, key=lambda p: p[0])
    mrp = compare_at if compare_at and compare_at > price else price
    return price, mrp, in_stock


def option_values(raw: dict, names: tuple[str, ...], available_only: bool = False) -> list[str]:
    options = raw.get("options") or []
    idx = next((i for i, o in enumerate(options) if str(o.get("name", "")).strip().lower() in names), None)
    if idx is None:
        return []
    values: list[str] = []
    for v in raw.get("variants") or []:
        if available_only and not v.get("available", True):
            continue
        opts = v.get("options") or []
        if idx < len(opts) and opts[idx] and opts[idx] not in values:
            values.append(str(opts[idx]))
    if not values and not available_only:
        values = [str(x) for x in options[idx].get("values", [])]
    return values


# --------------------------------------------------------------------- main
def normalize(raw: dict, site, scraped_at: str | None = None) -> dict | None:
    """Return a canonical product dict, or None if it isn't a priced fashion item."""
    scraped_at = scraped_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    title = WS.sub(" ", str(raw.get("title") or "")).strip()
    vendor = WS.sub(" ", str(raw.get("vendor") or raw.get("brand") or "")).strip()
    product_type = WS.sub(" ", str(raw.get("product_type") or "")).strip()
    raw_tags = [str(t) for t in (raw.get("tags") or [])]

    # Some stores put a style code in the title and the real name in vendor.
    model_code = None
    if looks_like_code(title) and len(vendor.split()) >= 3:
        model_code, title, vendor = title, vendor, ""
    if not title:
        return None

    description = strip_html(raw.get("description_html"))
    category, subcategory = infer_category(title, product_type, raw_tags)

    if not getattr(site, "include_non_fashion", False):
        # Skip homeware, gift cards, phone covers etc., even when a garment word
        # appears in the name (e.g. "Saree Gift Card").
        if category is None or NON_FASHION.search(f"{title} | {product_type}"):
            return None
    elif category is None:
        category, subcategory = "other", "other"

    price, mrp, in_stock = choose_price(raw.get("variants") or [])
    if price is None:
        return None
    discount = int(round((mrp - price) / mrp * 100)) if mrp and mrp > price else 0

    color_opts = option_values(raw, ("color", "colour"))
    colors = infer_colors(title, raw_tags, color_opts, raw.get("color"))
    sizes = option_values(raw, ("size",))
    available_sizes = option_values(raw, ("size",), available_only=True)
    gender = infer_gender([title, product_type, " ".join(raw_tags)], getattr(site, "default_gender", None))
    materials = infer_materials(title, " ".join(raw_tags), description)
    tags = clean_tags(raw_tags)

    brand = getattr(site, "brand", None) or nice_case(vendor) or site.name
    images = [img for img in (raw.get("images") or []) if img]
    images = ["https:" + img if img.startswith("//") else img for img in images]

    source_pid = str(raw.get("source_product_id") or raw.get("url"))
    product_id = f"{site.id}_{re.sub(r'[^A-Za-z0-9_.-]', '_', source_pid)}"

    embedding_text = " ".join(filter(None, [
        "" if brand.lower() in title.lower() else brand, title, category, subcategory, " ".join(colors), gender or "",
        " ".join(materials), product_type, " ".join(tags[:8]),
    ]))
    embedding_text = WS.sub(" ", embedding_text).strip()

    product = {
        "_id": product_id,
        "name": title,
        "category": category,
        "subcategory": subcategory or "",
        "product_type": product_type,
        "brand": brand,
        "gender": gender,
        "color": colors,
        "material": materials,
        "tags": tags,
        "price": price,
        "mrp": mrp,
        "discount_percent": discount,
        "currency": raw.get("currency") or getattr(site, "currency", "INR"),
        "rating": float(raw.get("rating") or 0.0),
        "review_count": int(raw.get("review_count") or 0),
        "in_stock": in_stock,
        "sizes": sizes,
        "available_sizes": available_sizes,
        "image_url": images[0] if images else "",
        "images": images[:5],
        "description": description,
        "embedding_text": embedding_text,
        "source": site.id,
        "source_name": site.name,
        "source_url": raw.get("url") or "",
        "source_product_id": source_pid,
        "scraped_at": scraped_at,
    }
    if model_code:
        product["model_code"] = model_code
    if raw.get("gtin"):
        product["gtin"] = str(raw["gtin"])
    return product
