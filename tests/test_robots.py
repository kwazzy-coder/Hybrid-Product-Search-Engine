from crawler.robots import RobotsPolicy

UA = "HybridSearchBot/1.0 (+me@example.com)"

# Trimmed copy of the robots.txt Shopify stores such as Libas and Powerlook serve.
SHOPIFY_ROBOTS = """
User-agent: *
Disallow: /admin
Disallow: /cart
Disallow: /checkout
Disallow: /collections/*sort_by*
Disallow: /*/collections/*sort_by*
Disallow: /collections/*+*
Disallow: /search
Disallow: /products/*-[a-f0-9][a-f0-9][a-f0-9][a-f0-9][a-f0-9][a-f0-9][a-f0-9][a-f0-9]-remote
Disallow: /*.atom
Sitemap: https://www.example.in/sitemap.xml

User-agent: AhrefsBot
Crawl-delay: 10
Disallow: /
"""


def test_shopify_rules():
    r = RobotsPolicy(SHOPIFY_ROBOTS)
    assert r.can_fetch(UA, "https://x.in/products.json?limit=250&page=2")
    assert r.can_fetch(UA, "https://x.in/products/blue-jeans")
    assert r.can_fetch(UA, "https://x.in/collections/men")
    assert not r.can_fetch(UA, "https://x.in/collections/men?sort_by=price-ascending")
    assert not r.can_fetch(UA, "https://x.in/collections/men+shirts")
    assert not r.can_fetch(UA, "https://x.in/search?q=kurta")
    assert not r.can_fetch(UA, "https://x.in/checkout")
    assert not r.can_fetch(UA, "https://x.in/collections/all.atom")
    assert r.sitemaps == ["https://www.example.in/sitemap.xml"]
    assert r.crawl_delay(UA) is None
    assert r.crawl_delay("AhrefsBot/7.0") == 10
    assert not r.can_fetch("AhrefsBot/7.0", "https://x.in/products.json")


def test_longest_match_and_allow_tiebreak():
    r = RobotsPolicy("User-agent: *\nDisallow: /folder\nAllow: /folder/page\nAllow: /p\nDisallow: /\n")
    assert r.can_fetch(UA, "https://x.in/folder/page")
    assert not r.can_fetch(UA, "https://x.in/folder/other")
    assert r.can_fetch(UA, "https://x.in/page")
    assert not r.can_fetch(UA, "https://x.in/zzz")
    tie = RobotsPolicy("User-agent: *\nDisallow: /a\nAllow: /a\n")
    assert tie.can_fetch(UA, "https://x.in/a")


def test_dollar_anchor_and_wildcards():
    r = RobotsPolicy("User-agent: *\nDisallow: /*.json$\n")
    assert not r.can_fetch(UA, "https://x.in/data.json")
    assert r.can_fetch(UA, "https://x.in/data.json?x=1")
    assert r.can_fetch(UA, "https://x.in/data.jsonl")


def test_specific_group_beats_star():
    r = RobotsPolicy("User-agent: *\nDisallow: /\n\nUser-agent: HybridSearchBot\nAllow: /\nCrawl-delay: 3\n")
    assert r.can_fetch(UA, "https://x.in/anything")
    assert r.crawl_delay(UA) == 3
    assert not r.can_fetch("OtherBot", "https://x.in/anything")


def test_grouped_user_agents_and_empty_disallow():
    r = RobotsPolicy("User-agent: GPTBot\nUser-agent: ClaudeBot\nDisallow: /\n\nUser-agent: *\nDisallow:\n")
    assert not r.can_fetch("ClaudeBot", "https://x.in/")
    assert r.can_fetch(UA, "https://x.in/products.json")


def test_missing_and_failing_robots():
    assert RobotsPolicy(allow_all=True).can_fetch(UA, "https://x.in/anything")
    assert not RobotsPolicy(disallow_all=True).can_fetch(UA, "https://x.in/anything")
