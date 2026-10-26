"""
Command-line entry point.

    python -m crawler check https://www.libas.in      # is this store crawlable? what does a product look like?
    python -m crawler crawl                           # crawl every enabled site in sites.toml
    python -m crawler crawl --site libas --max-products 200
    python -m crawler export                          # write data/products_live.json for the search service
    python -m crawler images                          # download thumbnails for CLIP embeddings
    python -m crawler refresh                         # crawl + export + images (use this from cron)
    python -m crawler stats                           # per-site counts and last run status
"""

from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace

from .adapters.shopify import NotShopify, ShopifyCrawl, probe
from .adapters.structured import SitemapCrawl, extract_product
from .config import CrawlerConfig, load_config
from .http import PoliteSession, RobotsDisallowed, SiteBlocked
from .images import download_images
from .normalize import normalize
from .store import CatalogStore, utcnow


def make_session(cfg: CrawlerConfig, use_cache: bool = True) -> PoliteSession:
    if not cfg.contact:
        print("Tip: set `contact` in crawler/sites.toml (or CRAWLER_CONTACT) so site owners can reach you.\n")
    return PoliteSession(
        cfg.full_user_agent,
        min_delay=cfg.min_delay_seconds,
        cache_dir=cfg.cache_dir if use_cache else None,
        cache_max_age_hours=cfg.cache_max_age_hours,
    )


# ------------------------------------------------------------------ crawl
def crawl_site(session: PoliteSession, store: CatalogStore, site, limit: int) -> dict:
    print(f"\n== {site.name} ({site.url})")
    run_id = store.start_run(site.id)
    saved = skipped = removed = new = price_changes = 0
    status, note = "ok", ""
    seen: set[str] = set()
    now = utcnow()

    platform = site.platform
    crawl = None
    try:
        if not session.allowed(site.url.rstrip("/") + "/products.json") and platform in ("shopify", "auto"):
            if platform == "shopify":
                raise RobotsDisallowed("/products.json")
            platform = "sitemap"
        if platform == "auto":
            platform = "shopify" if probe(session, site.url) else "sitemap"
            print(f"  detected platform: {platform}")
        crawl = (ShopifyCrawl if platform == "shopify" else SitemapCrawl)(session, site, limit)
        for raw in crawl:
            product = normalize(raw, site, now)
            if product is None:
                skipped += 1
                continue
            is_new, changed = store.upsert(product, now)
            new += is_new
            price_changes += changed
            seen.add(product["_id"])
            saved += 1
        store.commit()
    except RobotsDisallowed as exc:
        status, note = "disallowed", f"robots.txt disallows {exc}"
    except SiteBlocked as exc:
        status, note = "blocked", str(exc)
    except NotShopify as exc:
        status, note = "error", f"{exc} — set platform = \"sitemap\" for this site"
    except Exception as exc:  # keep going with the other sites
        status, note = "error", f"{type(exc).__name__}: {exc}"
    finally:
        store.commit()

    if status == "ok" and crawl is not None and crawl.complete:
        removed = store.mark_missing_inactive(site.id, seen)
    elif status == "ok" and crawl is not None and not crawl.complete:
        note = f"stopped at max_products={limit}; removals not checked"

    store.finish_run(run_id, status, saved, skipped, removed, note)
    print(f"  {status}: {saved} saved ({new} new, {price_changes} price changes), "
          f"{skipped} skipped (non-fashion or unpriced), {removed} no longer listed"
          + (f"\n  note: {note}" if note else ""))
    return {"site": site.id, "status": status, "saved": saved, "new": new, "skipped": skipped,
            "price_changes": price_changes, "removed": removed, "note": note}


def cmd_crawl(cfg: CrawlerConfig, args) -> int:
    sites = [s for s in cfg.sites if s.enabled]
    if args.site:
        sites = [s for s in cfg.sites if s.id in args.site]
        missing = set(args.site) - {s.id for s in sites}
        if missing:
            print(f"Unknown site id(s): {', '.join(sorted(missing))}")
            return 2
    if not sites:
        print("No sites to crawl. Add [[site]] entries to crawler/sites.toml.")
        return 2
    session = make_session(cfg, use_cache=not args.no_cache)
    store = CatalogStore(cfg.db_path)
    results = []
    for site in sites:
        limit = args.max_products or site.max_products or cfg.max_products_per_site
        results.append(crawl_site(session, store, site, limit))
    store.close()
    ok = sum(r["status"] == "ok" for r in results)
    print(f"\nDone: {ok}/{len(results)} sites crawled, {sum(r['saved'] for r in results)} products saved.")
    return 0 if ok else 1


def cmd_export(cfg: CrawlerConfig, args) -> int:
    store = CatalogStore(cfg.db_path)
    out = args.out or cfg.export_path
    count = store.export(out, include_inactive=args.include_inactive)
    store.close()
    print(f"Exported {count} products to {out}")
    return 0 if count else 1


def cmd_images(cfg: CrawlerConfig, args) -> int:
    store = CatalogStore(cfg.db_path)
    products = store.active_products()
    store.close()
    session = make_session(cfg, use_cache=False)
    session.set_host_delay("https://cdn.shopify.com", cfg.image_delay_seconds)
    print(f"Downloading thumbnails for {len(products)} products into {cfg.images_dir} ...")
    stats = download_images(session, products, cfg.images_dir, size=args.size or cfg.image_size,
                            force=args.force, limit=args.limit)
    print(f"Images: {stats}")
    return 0


def cmd_refresh(cfg: CrawlerConfig, args) -> int:
    code = cmd_crawl(cfg, args)
    cmd_export(cfg, SimpleNamespace(out=None, include_inactive=False))
    cmd_images(cfg, SimpleNamespace(size=None, force=False, limit=None))
    print("\nNext: cd search_service && python scripts/build_live_index.py")
    return code


def cmd_stats(cfg: CrawlerConfig, args) -> int:
    store = CatalogStore(cfg.db_path)
    for row in store.stats():
        run = row["last_run"] or {}
        print(f"{row['site']:<16} {row['active']:>6} active / {row['products']:>6} total   "
              f"{row['price_observations']:>6} price points   last run: {run.get('status')} {run.get('finished_at') or ''}"
              + (f"  ({run.get('note')})" if run.get("note") else ""))
    store.close()
    return 0


def cmd_check(cfg: CrawlerConfig, args) -> int:
    url = args.url.rstrip("/")
    session = make_session(cfg, use_cache=False)
    robots = session.robots_for(url)
    feed = url + "/products.json"
    print(f"Checking {url} as {cfg.full_user_agent}")
    print(f"  robots.txt sitemaps: {robots.sitemaps[:3] or 'none listed'}")
    print(f"  /products.json allowed by robots.txt: {robots.can_fetch(session.user_agent, feed)}")

    site = SimpleNamespace(id="check", name=url.split("//")[-1], url=url, brand=None,
                           default_gender=None, currency="INR", include_non_fashion=False)
    raw, platform = None, None
    try:
        if robots.can_fetch(session.user_agent, feed):
            raw = probe(session, url)
            platform = "shopify" if raw else None
        if raw is None and args.product_url:
            res = session.get(args.product_url, use_cache=False)
            raw = extract_product(res.text, args.product_url)
            platform = "sitemap" if raw else None
    except (RobotsDisallowed, SiteBlocked) as exc:
        print(f"  Not crawlable: {exc}")
        return 1

    if raw is None:
        print("  No Shopify feed found." + ("" if args.product_url else
              " If this isn't a Shopify store, re-run with --product-url <a product page> to test structured data."))
        return 1

    print(f"  Platform: {platform}")
    product = normalize(raw, site)
    print("  Sample product after normalization:")
    print(json.dumps(product, indent=2, ensure_ascii=False)[:2500] if product else "  (skipped: not a priced fashion item)")
    slug = url.split("//")[-1].split("/")[0].replace("www.", "").split(".")[0]
    print("\nAdd this to crawler/sites.toml:\n")
    print(f'[[site]]\nid = "{slug}"\nname = "{slug.capitalize()}"\nurl = "{url}"\nplatform = "{platform}"')
    if platform == "sitemap":
        print('product_url_pattern = "/products?/"   # adjust to match this site\'s product URLs')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m crawler", description="Collect real fashion products for the search engine.")
    parser.add_argument("--config", help="path to sites.toml (default: crawler/sites.toml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("crawl", help="crawl enabled sites")
    p.add_argument("--site", action="append", help="only this site id (repeatable)")
    p.add_argument("--max-products", type=int, help="per-site cap for this run")
    p.add_argument("--no-cache", action="store_true", help="ignore the HTTP cache")

    p = sub.add_parser("export", help="write the catalog JSON the search service reads")
    p.add_argument("--out")
    p.add_argument("--include-inactive", action="store_true")

    p = sub.add_parser("images", help="download product thumbnails for CLIP")
    p.add_argument("--size", type=int)
    p.add_argument("--limit", type=int)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("refresh", help="crawl + export + images")
    p.add_argument("--site", action="append")
    p.add_argument("--max-products", type=int)
    p.add_argument("--no-cache", action="store_true")

    sub.add_parser("stats", help="show what's been collected")

    p = sub.add_parser("check", help="test whether a store can be crawled")
    p.add_argument("url")
    p.add_argument("--product-url", help="a product page, for non-Shopify sites")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    handler = {"crawl": cmd_crawl, "export": cmd_export, "images": cmd_images,
               "refresh": cmd_refresh, "stats": cmd_stats, "check": cmd_check}[args.command]
    return handler(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
