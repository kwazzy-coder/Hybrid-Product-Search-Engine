# Fashion store crawler

Collects real products (names, prices, MRP, sizes, stock, colours, images and product links) from
Indian fashion brand websites and feeds them into the hybrid search engine. It also tracks
prices over time, so the UI can show price drops.

## Stores included

All five were checked on 26 Oct 2026: each publishes its catalog as public product data, and its
robots.txt allows crawlers to read it.

| id | Store | What they sell |
|---|---|---|
| `libas` | Libas (libas.in) | Women's ethnic: sarees, kurtas, kurta sets, dresses |
| `suta` | Suta (suta.in) | Handloom sarees and women's wear |
| `powerlook` | Powerlook (powerlook.in) | Menswear: jeans, shirts, t-shirts, cargos |
| `thebearhouse` | The Bear House (thebearhouse.com) | Menswear: shirts, sneakers |
| `bonkerscorner` | Bonkers Corner (bonkerscorner.com) | Streetwear: oversized tees, hoodies, caps, co-ords |

Rare Rabbit is left out on purpose: its robots.txt asks automated agents to use its official
commerce API instead. Myntra, Ajio, Amazon and Flipkart are not supported. Their terms forbid
scraping and they actively block bots, and this crawler doesn't try to get around blocking.

## Quick start

Requires Python 3.11+ (for `tomllib`).

```bash
pip install -r crawler/requirements.txt          # requests, Pillow

# 1. Put your email or GitHub URL in crawler/sites.toml -> contact (or set CRAWLER_CONTACT)

# 2. Crawl, export the catalog, download thumbnails (from the project root)
python -m crawler refresh                         # ~5 sites x up to 1000 products; takes a few minutes

# 3. Build the search indexes for the crawled catalog
cd search_service
python scripts/build_live_index.py                # BM25 + MiniLM + CLIP + cross-store matching

# 4. Start the API (it uses the live catalog automatically once step 3 has run)
uvicorn main:app --port 8000
```

Try a smaller first run: `python -m crawler refresh --max-products 100`.

To switch back to the original demo catalog, set `CATALOG=demo` (`set CATALOG=demo` on Windows
cmd, `$env:CATALOG="demo"` in PowerShell) before starting uvicorn.

## Commands

| Command | What it does |
|---|---|
| `python -m crawler check https://www.example.com` | Checks robots.txt, detects Shopify, shows a normalized sample product and prints a ready-made `[[site]]` block |
| `python -m crawler crawl [--site libas] [--max-products N] [--no-cache]` | Crawls sites into `data/crawler.db` |
| `python -m crawler export` | Writes `data/products_live.json` for the search service |
| `python -m crawler images [--limit N]` | Downloads 400px thumbnails to `data/images_live/` (used only for CLIP embeddings) |
| `python -m crawler refresh` | crawl + export + images |
| `python -m crawler stats` | Products per store, price observations, last run status |

## Keeping prices fresh

Run the refresh and the index build once a day. Price history builds up from the second run onward.

- **Linux/macOS (cron):** `30 6 * * * cd /path/to/project && python -m crawler refresh && cd search_service && python scripts/build_live_index.py`
- **Windows:** Task Scheduler → Create Basic Task → Daily → Start a program: `python`, arguments
  `-m crawler refresh`, start in: your project folder. Add a second action for the index build.

After rebuilding, restart the API so it loads the new catalog and indexes.

## How it behaves

- **robots.txt is checked before every request**, using the full RFC 9309 rules (wildcards and `$`),
  which Python's built-in parser doesn't support. A disallowed URL is never fetched.
- **2 seconds between requests to the same site** by default, longer if the site sets `Crawl-delay`.
- **Identifies itself** as `HybridSearchBot/1.0 (+your contact)`.
- **Backs off** on 429/503 (honouring `Retry-After`) and **stops** on 403 or repeated refusals. It
  never rotates IPs, spoofs browsers or solves CAPTCHAs.
- **Uses each store's public product feed** (Shopify's `/products.json`, 250 products per request)
  instead of loading thousands of pages. Non-Shopify stores are read from their sitemaps plus the
  schema.org Product data embedded in each page.
- **Caches responses for 12 hours**, so re-running during development doesn't hit the sites again.
- **Skips non-fashion items** (homeware, gift cards, phone covers) and products without a price.
- **Marks products as removed** when a complete crawl no longer finds them.

## What gets stored per product

On top of the fields the search engine already used (`name`, `category`, `price`, `mrp`,
`discount_percent`, `in_stock`, `color`, `tags`, `image_url`, `embedding_text`, `source`, `source_url`),
each product has `brand`, `gender`, `material`, `sizes`, `available_sizes`, `currency`, `description`,
`images`, `source_name`, `scraped_at`, `first_seen` and a `price_history` summary (`previous_price`,
`price_dropped`, `lowest_30d`, `series`).

Categories are normalized across stores (Libas's "Sarees", Suta's "Saree" and an untyped "Banarasi Silk
Saree" all become `saree`). Store quirks are handled too: The Bear House puts a style code in the title
and the product name in the vendor field, and Bonkers Corner leaves the product type empty.

## Responsible use

- This is for a personal or portfolio project. Read each store's terms before using the data for
  anything commercial.
- **Don't commit or republish the scraped data or images.** `data/products_live.json`, `data/crawler.db`,
  `data/images_live/` and `data/http_cache/` are in `.gitignore`. The UI shows images from the stores' own
  CDN and links every product back to the store.
- Keep the crawl rate low and the contact filled in. If a store asks you to stop, remove it from
  `sites.toml`.

## Tests

```bash
pip install pytest
python -m pytest tests -q
```

The tests run the crawler against local fake stores: pagination, robots.txt blocks, 429 back-off, 403
refusal, sitemap + JSON-LD extraction and price history across two crawls. The normalizer tests use the
shape of real listings from each store.
