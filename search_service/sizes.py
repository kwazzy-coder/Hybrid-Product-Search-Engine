"""
Clothing and shoe size handling.

Stores write sizes many ways ("XXL" / "2XL", "UK 8" / "8", "Free Size" / "Onesize"),
so everything is normalized before comparing.
"""

import re

ONE_SIZE = "ONE SIZE"
_ONE_SIZE_NAMES = {
    "FREE SIZE", "FREESIZE", "FREE", "ONESIZE", "ONE SIZE", "ONE-SIZE", "OS", "STANDARD",
    "DEFAULT TITLE", "DEFAULT", "NA", "N/A", "UNSTITCHED",
}
_ALIASES = {
    "2XL": "XXL", "3XL": "XXXL", "4XL": "XXXXL", "XXS": "XXS", "SMALL": "S", "MEDIUM": "M", "LARGE": "L",
    "XSMALL": "XS", "X-SMALL": "XS", "XLARGE": "XL", "X-LARGE": "XL", "EXTRA LARGE": "XL", "EXTRALARGE": "XL",
}
LETTER_ORDER = ["XXS", "XS", "S", "M", "L", "XL", "XXL", "XXXL", "XXXXL"]


def normalize_size(value) -> str | None:
    if value is None:
        return None
    s = re.sub(r"\s+", " ", str(value)).strip().upper()
    if not s:
        return None
    if s in _ONE_SIZE_NAMES:
        return ONE_SIZE
    s = re.sub(r"^(?:UK|IND|IN)\s*-?\s*(?=\d)", "", s)   # "UK 8" -> "8" (Indian sizing follows UK)
    s = re.sub(r"(?<=\d)\s*(?:UK|IND)$", "", s)            # "8 UK" -> "8"
    s = _ALIASES.get(s, s)
    return s.replace(" ", "")


def size_sort_key(size: str):
    if size in LETTER_ORDER:
        return (0, LETTER_ORDER.index(size), "")
    if size.isdigit():
        return (1, int(size), "")
    return (2, 0, size)


def size_matches(product: dict, size: str | None) -> bool:
    """Is the product available in `size`? One-size items and items without size data always match."""
    if not size:
        return True
    want = normalize_size(size)
    sizes = product.get("sizes") or []
    if not sizes:
        return True
    all_sizes = {normalize_size(s) for s in sizes}
    if all_sizes <= {ONE_SIZE, None}:
        return True
    available = product.get("available_sizes")
    pool = available if available is not None else sizes
    return want in {normalize_size(s) for s in pool}


# --- query parsing ------------------------------------------------------
_LETTER = r"xxs|xs|s|m|l|xl|xxl|xxxl|[2-4]xl"
_SIZE_PATTERNS = [
    re.compile(rf"\bsize\s*[:\-]?\s*({_LETTER}|\d{{1,2}})\b"),        # "size L", "size 32"
    re.compile(rf"\b({_LETTER}|\d{{1,2}})\s+size\b"),                  # "M size kurta"
    re.compile(r"\b(?:uk|ind)\s*-?\s*(\d{1,2})\b"),                     # "UK 8 sneakers"
    re.compile(r"\b(\d{2})\s*(?:inch\s+)?waist\b"),                     # "32 waist jeans"
    re.compile(r"\b(xxs|xs|xl|xxl|xxxl|[2-4]xl)\b"),                   # standalone, unambiguous letters
]


def extract_size(query: str) -> tuple[str | None, str]:
    """Return (normalized size or None, query with the size phrase removed)."""
    lowered = query.lower()
    for pattern in _SIZE_PATTERNS:
        m = pattern.search(lowered)
        if m:
            cleaned = (lowered[:m.start()] + " " + lowered[m.end():]).strip()
            return normalize_size(m.group(1)), re.sub(r"\s+", " ", cleaned)
    return None, lowered
