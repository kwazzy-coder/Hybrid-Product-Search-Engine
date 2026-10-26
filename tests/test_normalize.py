"""
Normalizer tests. The raw products copy the shape and values of real listings
seen on each store's /products.json (Oct 2026).
"""

from types import SimpleNamespace

from crawler.normalize import infer_category, normalize


def site(**kw):
    base = dict(id="store", name="Store", url="https://store.in", brand=None, default_gender=None,
                currency="INR", include_non_fashion=False)
    base.update(kw)
    return SimpleNamespace(**base)


def variant(price, compare=None, available=True, opts=(None, None, None)):
    return {"price": price, "compare_at_price": compare, "available": available, "options": list(opts)}


def test_libas_saree():
    raw = {
        "source_product_id": "7741", "url": "https://www.libas.in/products/beige-saree",
        "title": "Libas Beige and Green Woven Design Silk Blend Saree", "vendor": "Libas",
        "product_type": "Sarees",
        "tags": ["1104128P", "AW26", "AW26 SAREE", "Beige Sarees", "EXCLA", "Floral Sarees", "Party Wear Sarees",
                 "Silk Sarees", "NEWARRIVALSAW26", "Woven Design Sarees"],
        "images": ["https://cdn.shopify.com/s/files/1/0341/files/saree.jpg?v=1"],
        "options": [{"name": "Size", "values": ["Onesize"]}],
        "variants": [variant("2999.00", "2999.00", True, ("Onesize", None, None))],
    }
    p = normalize(raw, site(id="libas", name="Libas", brand="Libas", default_gender="women"))
    assert p["_id"] == "libas_7741"
    assert p["category"] == "saree" and p["subcategory"] == "ethnicwear"
    assert p["price"] == 2999.0 and p["mrp"] == 2999.0 and p["discount_percent"] == 0
    assert {"beige", "green"} <= set(p["color"])
    assert p["gender"] == "women"
    assert "silk blend" in p["material"]
    assert "aw26" not in p["tags"] and "newarrivalsaw26" not in p["tags"] and "party wear sarees" in p["tags"]
    assert p["source_url"] == "https://www.libas.in/products/beige-saree"
    assert "saree" in p["embedding_text"].lower()


def test_suta_title_without_garment_word_uses_product_type():
    raw = {
        "source_product_id": "1", "url": "https://suta.in/products/sheetal-safed", "title": "Sheetal Safed",
        "vendor": "suta.in", "product_type": "Saree",
        "tags": ["Colour_White", "One Of A Kind", "Pre Draped", "Saree"],
        "options": [{"name": "Title", "values": ["Default Title"]}],
        "variants": [variant("2100.00", None, False)],
    }
    p = normalize(raw, site(id="suta", name="Suta", brand="Suta", default_gender="women"))
    assert p["category"] == "saree"
    assert p["color"] == ["white"]
    assert p["in_stock"] is False
    assert p["brand"] == "Suta"


def test_powerlook_jeans_with_vague_product_type():
    raw = {
        "source_product_id": "2", "url": "https://www.powerlook.in/products/blue-wash-baggy-fit-jeans",
        "title": "Blue Washed Baggy Fit Jeans", "vendor": "powerlook-apparels", "product_type": "Bottom",
        "tags": ["All Products", "Bottoms", "Jeans & Cargo", "New Arrival", "Not For Sale"],
        "options": [{"name": "Size", "values": ["28", "30", "32"]}, {"name": "Color", "values": ["Blue"]}],
        "variants": [variant("1899.00", None, True, ("28", "Blue", None)),
                     variant("1899.00", None, False, ("30", "Blue", None)),
                     variant("1899.00", None, True, ("32", "Blue", None))],
    }
    p = normalize(raw, site(id="powerlook", name="Powerlook", brand="Powerlook", default_gender="men"))
    assert p["category"] == "jeans" and p["subcategory"] == "bottomwear"
    assert p["sizes"] == ["28", "30", "32"] and p["available_sizes"] == ["28", "32"]
    assert "blue" in p["color"]
    assert p["gender"] == "men"


def test_bear_house_code_in_title_is_swapped_with_vendor():
    raw = {
        "source_product_id": "3", "url": "https://thebearhouse.com/products/bearsky-rd",
        "title": "BEARSKY-RD", "vendor": "Red PU Lace-Up Casual Sneakers with Contrast Gold BEAR Detail",
        "product_type": "Shoes And Loafers",
        "tags": ["64041110", "Collection_Smart_Casual", "Lace Up", "made in india", "Red", "Shoes", "Sneakers"],
        "options": [{"name": "Size", "values": ["UK7", "UK8"]}],
        "variants": [variant("2599.00", "2599.00", True, ("UK7", None, None))],
    }
    p = normalize(raw, site(id="thebearhouse", name="The Bear House", brand="The Bear House", default_gender="men"))
    assert p["name"].startswith("Red PU Lace-Up Casual Sneakers")
    assert p["model_code"] == "BEARSKY-RD"
    assert p["category"] == "shoes" and p["subcategory"] == "footwear"
    assert p["color"][0] == "red"
    assert p["brand"] == "The Bear House"


def test_bonkers_cap_empty_type_and_discount():
    raw = {
        "source_product_id": "4", "url": "https://www.bonkerscorner.com/products/bad-hair-day-cap",
        "title": "Bad Hair Day Cap", "vendor": "BonkersCorner store", "product_type": "",
        "tags": ["Men", "New In", "Non-Returnable", "Women"],
        "options": [{"name": "Size", "values": ["Free Size"]}],
        "variants": [variant("799.00", "1099.00", True, ("Free Size", None, None))],
    }
    p = normalize(raw, site(id="bonkerscorner", name="Bonkers Corner", brand="Bonkers Corner"))
    assert p["category"] == "cap" and p["subcategory"] == "accessories"
    assert p["price"] == 799 and p["mrp"] == 1099 and p["discount_percent"] == 27
    assert p["gender"] == "unisex"


def test_non_fashion_and_unpriced_are_skipped():
    dinner = {"source_product_id": "5", "title": "Haathi Dinner Set - Set of 2", "vendor": "Nicobar",
              "product_type": "shop-bundle-quantity", "tags": ["Filtercategory_House & Home", "MATERIAL_Stoneware"],
              "variants": [variant("8400.00")]}
    gift = {"source_product_id": "6", "title": "Saree Gift Card", "product_type": "Gift Card",
            "variants": [variant("1000.00")]}
    unpriced = {"source_product_id": "7", "title": "Linen Shirt", "variants": [variant(None)]}
    assert normalize(dinner, site()) is None
    assert normalize(gift, site()) is None
    assert normalize(unpriced, site()) is None


def test_mrp_uses_cheapest_available_variant():
    raw = {"source_product_id": "8", "title": "Cotton Kurta", "variants": [
        variant("1500", "2000", False), variant("1200", "1999", True), variant("1300", None, True)]}
    p = normalize(raw, site())
    assert (p["price"], p["mrp"], p["discount_percent"], p["in_stock"]) == (1200, 1999, 40, True)


def test_category_head_noun_rules():
    cases = {
        "Women Navy Blue Printed Kurta with Trousers & Dupatta": "kurta set",
        "Banarasi Silk Saree with Blouse Piece": "saree",
        "Striped Shirt Dress": "dress",
        "Cap Sleeve Top": "top",
        "Oversized Hooded Sweatshirt": "hoodie",
        "Men Solid Polo T-Shirt": "tshirt",
        "Relaxed Linen Shirt": "shirt",
        "Cotton Co-ord Set": "co-ord set",
        "Straight Fit Cargo Pants": "trousers",
        "Embroidered Juttis": "shoes",
        "Mirror Work Potli Bag": "bag",
    }
    for title, expected in cases.items():
        assert infer_category(title, "", [])[0] == expected, title


def test_generic_product_type_does_not_beat_specific_title():
    assert infer_category("Oversized T-Shirt", "Top", [])[0] == "tshirt"
    assert infer_category("Relaxed Linen Shirt", "Top", [])[0] == "shirt"
    assert infer_category("Ribbed Crop Top", "Top", [])[0] == "top"
    assert infer_category("Straight Fit Cargo Pants", "Bottom", [])[0] == "trousers"
    # A specific merchant label still wins over a misleading title word
    assert infer_category("Sheetal Safed", "Saree", [])[0] == "saree"
    assert infer_category("Floral Print Shirt", "SHIRT", [])[0] == "shirt"
