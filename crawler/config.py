"""Load crawler settings and the list of sites from sites.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "crawler", "sites.toml")


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


@dataclass
class SiteConfig:
    id: str
    name: str
    url: str
    platform: str = "auto"              # "shopify", "sitemap", or "auto" (detect)
    brand: str | None = None            # force a brand name (single-brand stores)
    default_gender: str | None = None   # "men" / "women" when the store is one-sided
    currency: str = "INR"
    collections: list[str] = field(default_factory=list)   # Shopify: limit to these collections
    sitemaps: list[str] = field(default_factory=list)      # sitemap sites: explicit sitemap URLs
    product_url_pattern: str | None = None                 # regex a product URL must match
    max_products: int | None = None
    include_non_fashion: bool = False
    enabled: bool = True


@dataclass
class CrawlerConfig:
    user_agent: str = "HybridSearchBot/1.0"
    contact: str = ""
    min_delay_seconds: float = 2.0
    image_delay_seconds: float = 0.5
    max_products_per_site: int = 1000
    cache_dir: str = "data/http_cache"
    cache_max_age_hours: float = 12.0
    db_path: str = "data/crawler.db"
    export_path: str = "data/products_live.json"
    images_dir: str = "data/images_live"
    image_size: int = 400
    alerts_db: str = "data/feedback.db"
    alerts_outbox: str = "data/alerts_outbox.txt"
    sites: list[SiteConfig] = field(default_factory=list)

    @property
    def full_user_agent(self) -> str:
        if self.contact:
            return f"{self.user_agent} (+{self.contact})"
        return self.user_agent

    def site(self, site_id: str) -> SiteConfig | None:
        return next((s for s in self.sites if s.id == site_id), None)


def load_config(path: str | None = None) -> CrawlerConfig:
    path = path or os.getenv("CRAWLER_CONFIG") or DEFAULT_CONFIG
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    settings = raw.get("crawler", {})
    cfg = CrawlerConfig(**{k: v for k, v in settings.items() if k in CrawlerConfig.__dataclass_fields__ and k != "sites"})
    # Environment overrides so the contact never has to be committed.
    cfg.contact = os.getenv("CRAWLER_CONTACT", cfg.contact)

    # FEEDBACK_DB is the same override the search service uses for this file.
    cfg.alerts_db = os.getenv("FEEDBACK_DB", cfg.alerts_db)
    for path_field in ("cache_dir", "db_path", "export_path", "images_dir", "alerts_db", "alerts_outbox"):
        setattr(cfg, path_field, _resolve(getattr(cfg, path_field)))

    for entry in raw.get("site", []):
        known = {k: v for k, v in entry.items() if k in SiteConfig.__dataclass_fields__}
        cfg.sites.append(SiteConfig(**known))
    return cfg
