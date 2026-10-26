import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "search_service"))

from sizes import extract_size, normalize_size, size_matches, size_sort_key  # noqa: E402


def test_normalize():
    assert normalize_size("2XL") == normalize_size("xxl") == "XXL"
    assert normalize_size("UK 8") == normalize_size("8 UK") == normalize_size("8") == "8"
    assert normalize_size("Free Size") == normalize_size("Onesize") == normalize_size("Default Title") == "ONE SIZE"
    assert normalize_size("medium") == "M"


def test_extract_size_from_queries():
    assert extract_size("black shirt size L") == ("L", "black shirt")
    assert extract_size("M size kurta") == ("M", "kurta")
    assert extract_size("jeans 32 waist") == ("32", "jeans")
    assert extract_size("white sneakers uk 8") == ("8", "white sneakers")
    assert extract_size("oversized tee in xxl")[0] == "XXL"
    # Ambiguous single letters / prices are not sizes
    assert extract_size("kurta set under 2000") == (None, "kurta set under 2000")
    assert extract_size("m&s style shirt")[0] is None


def test_size_matching():
    shirt = {"sizes": ["S", "M", "L", "XL"], "available_sizes": ["M", "XL"]}
    assert size_matches(shirt, "M") and size_matches(shirt, "xl")
    assert not size_matches(shirt, "L")
    saree = {"sizes": ["Onesize"], "available_sizes": ["Onesize"]}
    assert size_matches(saree, "M")                     # one-size fits everyone
    assert size_matches({"name": "demo item"}, "M")      # no size data: don't exclude
    sneaker = {"sizes": ["UK7", "UK8", "UK9"], "available_sizes": ["UK8"]}
    assert size_matches(sneaker, "8") and not size_matches(sneaker, "9")


def test_sort_order():
    assert sorted(["XL", "32", "S", "M", "28", "XXL"], key=size_sort_key) == ["S", "M", "XL", "XXL", "28", "32"]
