"""
Builds the photo-search index in a background thread after the API starts (lite mode).

On a free host the crawl runs at every cold start, so the image index has to be built
there too. Doing it in the background means the site is usable straight away: text
search works immediately, and photo search switches on as soon as the first photos
are indexed, covering more of the catalog as it goes. Stores are interleaved so a
partial index covers every store, not just the first one.

Each product's thumbnail comes from (in order):
  1. the demo catalog's own image files (search_service/static/...),
  2. a thumbnail `python -m crawler images` already saved (data/images_live/),
  3. the store's image CDN, through the crawler's polite session (robots.txt,
     per-host delay, back-off), asking Shopify for a small 256px version.
Images are only held in memory long enough to embed them.

Settings:
    PHOTO_INDEX_MAX   index at most this many products (default: all)
"""

from __future__ import annotations

import io
import os
import sys
import threading
import time
from collections import defaultdict
from itertools import zip_longest

from paths import PROJECT_DIR, SERVICE_DIR

THUMB_SIZE = 256
BATCH_SIZE = 4
SAVE_EVERY = 48


def interleave_by_store(products: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in products:
        groups[p.get("source", "unknown")].append(p)
    return [p for row in zip_longest(*groups.values()) for p in row if p is not None]


class ImageFetcher:
    """Gets a product image as a PIL image, from disk if possible, else politely from the store's CDN."""

    def __init__(self):
        self.session = None
        self.images_dir = os.path.join(PROJECT_DIR, "data", "images_live")
        self._thumbnail_url = lambda url, size: url
        self._safe_name = None
        self._blocked: set[str] = set()
        try:
            if PROJECT_DIR not in sys.path:
                sys.path.insert(0, PROJECT_DIR)
            from crawler.config import load_config
            from crawler.http import PoliteSession
            from crawler.images import safe_name, thumbnail_url

            cfg = load_config()
            self.images_dir = cfg.images_dir
            self.session = PoliteSession(cfg.full_user_agent, min_delay=cfg.image_delay_seconds, verbose=False)
            self._thumbnail_url = thumbnail_url
            self._safe_name = safe_name
        except Exception as exc:  # crawler package missing or misconfigured: local files only
            print(f"Photo index: can't download store images ({type(exc).__name__}: {exc}); using local files only.")

    def _open(self, data_or_path):
        from PIL import Image

        src = io.BytesIO(data_or_path) if isinstance(data_or_path, bytes) else data_or_path
        img = Image.open(src)
        img.draft("RGB", (THUMB_SIZE, THUMB_SIZE))  # JPEG: decode at reduced size, saves memory and time
        return img.convert("RGB")

    def fetch(self, product: dict):
        url = product.get("image_url") or ""
        if url.startswith("/static/"):
            path = os.path.join(SERVICE_DIR, url.lstrip("/"))
            return self._open(path) if os.path.exists(path) else None
        if self._safe_name:
            path = os.path.join(self.images_dir, self._safe_name(product["_id"]))
            if os.path.exists(path):
                return self._open(path)
        if not url.startswith("http") or self.session is None:
            return None
        from urllib.parse import urlsplit

        host = urlsplit(url).netloc
        if host in self._blocked:
            return None
        try:
            data = self.session.get_bytes(self._thumbnail_url(url, THUMB_SIZE))
        except Exception as exc:  # RobotsDisallowed / SiteBlocked: stop asking this host
            print(f"Photo index: images from {host} unavailable ({type(exc).__name__}); skipping that store's photos.")
            self._blocked.add(host)
            return None
        return self._open(data) if data else None


class PhotoIndexer:
    def __init__(self, retriever, fetcher_factory=ImageFetcher):
        self.retriever = retriever
        self.fetcher_factory = fetcher_factory
        self.state = "idle"        # idle | loading_model | indexing | ready | error
        self.total = 0
        self.failed = 0
        self.error = None
        self.seconds_per_photo = None
        self._thread = None

    @property
    def indexed(self) -> int:
        return int(self.retriever.index.ntotal)

    @property
    def building(self) -> bool:
        return self.state in ("loading_model", "indexing")

    @property
    def model_ready(self) -> bool:
        return self.retriever.model is not None

    def status(self) -> dict:
        return {
            "state": self.state, "building": self.building, "indexed": self.indexed, "total": self.total,
            "failed": self.failed, "error": self.error,
            "seconds_per_photo": round(self.seconds_per_photo, 2) if self.seconds_per_photo else None,
        }

    def describe(self) -> str:
        if self.state == "loading_model":
            return "getting ready: loading the image model"
        if self.state == "indexing":
            return f"getting ready: {self.indexed:,} of {self.total:,} product photos indexed"
        if self.state == "error":
            return self.error or "the photo index failed to build"
        return ""

    def start(self, products: list[dict]):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.run, args=(products,), name="photo-indexer", daemon=True)
        self._thread.start()

    def run(self, products: list[dict]):
        try:
            self.state = "loading_model"
            self.retriever._load_model()
            if self.retriever.model is None:
                self.state, self.error = "error", f"image model failed to load: {self.retriever.load_error}"
                print(f"Photo search unavailable: {self.error}")
                return

            limit = int(os.getenv("PHOTO_INDEX_MAX", "0") or 0)
            done = set(self.retriever.id_map.values())  # from an index saved by an earlier run
            todo = [p for p in interleave_by_store(products) if p.get("image_url") and p["_id"] not in done]
            if limit:
                todo = todo[:max(0, limit - len(done))]
            self.total = len(done) + len(todo)
            self.state = "indexing"
            if todo:
                print(f"Photo index: {len(done)} products already indexed, embedding {len(todo)} more in the background.")
            fetcher = self.fetcher_factory()
            started, embedded, since_save = time.perf_counter(), 0, 0
            batch_imgs, batch_ids = [], []

            def flush():
                nonlocal embedded, since_save
                if not batch_imgs:
                    return
                t0 = time.perf_counter()
                emb = self.retriever.encode_images(batch_imgs)
                per = (time.perf_counter() - t0) / len(batch_imgs)
                self.seconds_per_photo = per if self.seconds_per_photo is None else 0.8 * self.seconds_per_photo + 0.2 * per
                self.retriever.add_image_embeddings(emb, list(batch_ids), verbose=False)
                embedded += len(batch_ids)
                since_save += len(batch_ids)
                batch_imgs.clear()
                batch_ids.clear()
                if since_save >= SAVE_EVERY:
                    self.retriever.save()
                    since_save = 0

            for product in todo:
                try:
                    img = fetcher.fetch(product)
                except Exception:
                    img = None
                if img is None:
                    self.failed += 1
                    continue
                batch_imgs.append(img)
                batch_ids.append(product["_id"])
                if len(batch_imgs) >= BATCH_SIZE:
                    flush()
            flush()
            if embedded:
                self.retriever.save()
            self.state = "ready"
            print(f"Photo index ready: {self.indexed} products ({embedded} embedded in "
                  f"{time.perf_counter() - started:.0f}s, {self.failed} without a usable image).")
        except Exception as exc:
            self.state, self.error = "error", f"{type(exc).__name__}: {exc}"
            print(f"Photo index build failed: {self.error}")
