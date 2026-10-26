import gzip
import json

from crawler.adapters.structured import extract_product, parse_sitemap


def page(*jsonld_blocks, meta=""):
    scripts = "".join(f'<script type="application/ld+json">{b}</script>' for b in jsonld_blocks)
    return f"<html><head>{meta}{scripts}</head><body><h1>x</h1></body></html>"


def test_plain_product_with_offer():
    block = json.dumps({
        "@context": "https://schema.org", "@type": "Product", "name": "Women Black Printed Kurta",
        "sku": "BK-101", "brand": {"@type": "Brand", "name": "Anouk"},
        "image": ["https://img.example.com/1.jpg", "https://img.example.com/2.jpg"],
        "color": "Black",
        "offers": {"@type": "Offer", "price": "899", "priceCurrency": "INR",
                   "availability": "https://schema.org/InStock"},
        "aggregateRating": {"@type": "AggregateRating", "ratingValue": "4.3", "reviewCount": "128"},
    })
    raw = extract_product(page(block), "https://x.in/p/kurta")
    assert raw["title"] == "Women Black Printed Kurta"
    assert raw["brand"] == "Anouk" and raw["source_product_id"] == "BK-101"
    assert raw["variants"][0]["price"] == "899" and raw["variants"][0]["available"] is True
    assert raw["rating"] == 4.3 and raw["review_count"] == 128 and raw["currency"] == "INR"
    assert raw["images"][0] == "https://img.example.com/1.jpg"


def test_graph_aggregate_offer_and_list_price():
    block = json.dumps({"@context": "https://schema.org", "@graph": [
        {"@type": "WebPage", "name": "page"},
        {"@type": ["Product"], "name": "Denim Jacket", "productID": "DJ1",
         "offers": {"@type": "AggregateOffer", "lowPrice": "1999", "offers": [
             {"@type": "Offer", "price": "1999", "availability": "OutOfStock", "size": "M",
              "priceSpecification": [{"@type": "UnitPriceSpecification", "price": "2999",
                                      "priceType": "https://schema.org/StrikethroughPrice"}]},
             {"@type": "Offer", "price": "2099", "availability": "InStock", "size": "L"}]}},
    ]})
    raw = extract_product(page(block), "https://x.in/p/jacket")
    assert raw["title"] == "Denim Jacket"
    assert [v["price"] for v in raw["variants"]] == ["1999", "2099"]
    assert raw["variants"][0]["compare_at_price"] == "2999"
    assert raw["variants"][0]["available"] is False and raw["variants"][1]["available"] is True
    assert raw["options"] == [{"name": "Size", "values": ["L", "M"]}]


def test_product_group_with_variants_and_trailing_comma():
    block = """{"@context":"https://schema.org","@type":"ProductGroup","name":"Linen Shirt","productGroupID":"LS",
      "hasVariant":[{"@type":"Product","size":"S","offers":{"@type":"Offer","price":1499,"availability":"InStock"}},
                    {"@type":"Product","size":"M","offers":{"@type":"Offer","price":1499,"availability":"InStock"}},],}"""
    raw = extract_product(page(block), "https://x.in/p/shirt")
    assert raw["title"] == "Linen Shirt" and len(raw["variants"]) == 2


def test_open_graph_fallback_and_no_data():
    meta = ('<meta property="og:title" content="Graphic Tee"><meta property="og:image" content="https://i/1.jpg">'
            '<meta property="product:price:amount" content="599"><meta property="product:price:currency" content="INR">'
            '<meta property="product:original_price:amount" content="999">')
    raw = extract_product(page(meta=meta), "https://x.in/p/tee")
    assert raw["title"] == "Graphic Tee" and raw["variants"][0]["price"] == "599"
    assert raw["variants"][0]["compare_at_price"] == "999"
    assert extract_product("<html><head><title>About us</title></head></html>", "https://x.in/about") is None


def test_sitemap_index_urlset_and_gzip():
    index = b"""<?xml version="1.0"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>https://x.in/sitemap_products_1.xml?from=1&amp;to=9</loc></sitemap></sitemapindex>"""
    urlset = b"""<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://x.in/p/a</loc></url><url><loc>https://x.in/p/b</loc></url></urlset>"""
    assert parse_sitemap(index) == (["https://x.in/sitemap_products_1.xml?from=1&to=9"], [])
    assert parse_sitemap(gzip.compress(urlset)) == ([], ["https://x.in/p/a", "https://x.in/p/b"])
    assert parse_sitemap(b"not xml") == ([], [])
