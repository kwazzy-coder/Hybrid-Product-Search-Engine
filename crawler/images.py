"""
Download a small thumbnail per product so the search service can build CLIP
image embeddings offline. The frontend keeps showing the store's own CDN image;
these local copies are only used for embeddings.
"""

from __future__ import annotations

import io
import os
import re
from urllib.parse import urlsplit, urlencode, parse_qsl, urlunsplit

from .http import PoliteSession, RobotsDisallowed, SiteBlocked


def safe_name(product_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", product_id) + ".jpg"


def thumbnail_url(url: str, size: int) -> str:
    """Ask Shopify's CDN for a resized image instead of the full-resolution original."""
    parts = urlsplit(url)
    if "cdn.shopify.com" in parts.netloc or "/cdn/shop/" in parts.path:
        query = dict(parse_qsl(parts.query))
        query["width"] = str(size)
        return urlunsplit((parts.scheme or "https", parts.netloc, parts.path, urlencode(query), ""))
    return url


def download_images(session: PoliteSession, products: list[dict], out_dir: str, size: int = 400,
                    force: bool = False, limit: int | None = None) -> dict:
    from PIL import Image  # imported here so the crawler itself doesn't need Pillow

    os.makedirs(out_dir, exist_ok=True)
    stats = {"downloaded": 0, "skipped_existing": 0, "failed": 0, "no_image": 0}
    blocked_hosts: set[str] = set()

    for product in products:
        if limit is not None and stats["downloaded"] >= limit:
            break
        url = product.get("image_url")
        if not url:
            stats["no_image"] += 1
            continue
        path = os.path.join(out_dir, safe_name(product["_id"]))
        if os.path.exists(path) and not force:
            stats["skipped_existing"] += 1
            continue
        host = urlsplit(url).netloc
        if host in blocked_hosts:
            stats["failed"] += 1
            continue
        try:
            data = session.get_bytes(thumbnail_url(url, size))
        except (RobotsDisallowed, SiteBlocked) as exc:
            session.log(f"  images from {host} unavailable: {exc}")
            blocked_hosts.add(host)
            stats["failed"] += 1
            continue
        if not data:
            stats["failed"] += 1
            continue
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img.thumbnail((size, size))
            img.save(path, "JPEG", quality=85)
            stats["downloaded"] += 1
        except Exception:
            stats["failed"] += 1
        if stats["downloaded"] and stats["downloaded"] % 100 == 0:
            session.log(f"  {stats['downloaded']} images downloaded")
    return stats
