"""
Decide which catalog and which index folder the search service uses.

    CATALOG=live   use data/products_live.json (crawled) + models_live/
    CATALOG=demo   use the original synthetic data/products_2k.json + models/
    CATALOG=auto   (default) live if the crawled catalog and its indexes exist, else demo
"""

import os

SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SERVICE_DIR)
DATA_DIRS = [d for d in (os.getenv("DATA_DIR"), "/data", os.path.join(PROJECT_DIR, "data")) if d]

LEGACY_INDEX_DIR = os.path.join(SERVICE_DIR, "models")
LIVE_INDEX_DIR = os.getenv("LIVE_INDEX_DIR", os.path.join(SERVICE_DIR, "models_live"))


def find_data_file(*names: str) -> str | None:
    for directory in DATA_DIRS:
        for name in names:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                return path
    return None


LIVE_CATALOG = find_data_file("products_live.json")
LEGACY_CATALOG = find_data_file("products_2k.json", "products_100k_backup.json")

_requested = os.getenv("CATALOG", "auto").strip().lower()
_live_ready = bool(LIVE_CATALOG) and os.path.exists(os.path.join(LIVE_INDEX_DIR, "bm25.pkl"))

if _requested == "live" or (_requested == "auto" and _live_ready):
    CATALOG_MODE = "live"
    CATALOG_PATH = LIVE_CATALOG
    INDEX_DIR = LIVE_INDEX_DIR
else:
    CATALOG_MODE = "demo"
    CATALOG_PATH = LEGACY_CATALOG
    INDEX_DIR = LEGACY_INDEX_DIR

MATCHES_PATH = os.path.join(INDEX_DIR, "matches.json")
os.makedirs(INDEX_DIR, exist_ok=True)
