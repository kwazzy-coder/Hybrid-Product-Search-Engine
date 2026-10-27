# Deploying a public demo

The whole app runs as **one container**: the built React site and the FastAPI search API on
one URL. There are two builds:

| Build | File | What it includes | Memory |
|---|---|---|---|
| **Lite** | `Dockerfile.lite` | BM25 search, LightGBM re-ranking, filters, size filter, price history, photo search (CLIP image model on ONNX Runtime) | ~350 MB (fits 512 MB free tiers); ~150 MB with `PHOTO_SEARCH=off` |
| **Full** | `Dockerfile` | Everything in lite, plus MiniLM semantic search and CLIP photo search | ~2 GB |

## Before you publish: the data

A public demo shows the stores' product names, prices and images (loaded from their CDN) to
anyone with the link. The crawler respects robots.txt, but **check each store's terms of use
before putting their catalog on a public site.** If you'd rather not, there are safer options:

- deploy with the original synthetic catalog (`CATALOG=demo`), and show the live version in a
  short screen recording on your README; or
- keep the live demo for interviews only and don't link it publicly.

Nothing scraped is committed to the repo either way: `data/products_live.json`, `data/crawler.db`
and `data/images_live/` are in `.gitignore`. With `CRAWL_ON_START=1` the container builds its own
small catalog when it starts.

## Option 1: Render (free, lite build)

Render's free web services have 512 MB of memory and 0.1 CPU, and sleep after 15 minutes without
traffic (the first visit after that takes about a minute). The lite build fits.

1. Push the repo to GitHub.
2. In Render: **New → Blueprint**, pick the repo. It reads `render.yaml`.
3. When asked for `CRAWLER_CONTACT`, enter your email or GitHub URL.
4. Deploy. On each start the container crawls up to 200 products per store, builds the index
   and starts serving. The URL is shown on the service page.

Price alerts are switched off (`ALERTS_ENABLED=false`): a public form that sends email to any
address can be abused, and the free tier has no persistent disk to keep alerts on anyway.

## Option 2: Hugging Face Spaces (full build, paid)

Since mid-2026, creating a Docker Space requires a Hugging Face PRO account (free accounts are
limited to static Spaces and up to two Gradio apps on ZeroGPU). With PRO, the "CPU Basic"
hardware (2 vCPU, 16 GB, no hourly charge) runs the full build comfortably.

1. Create a Space with the **Docker** SDK.
2. Push this repo to the Space, using `deploy/huggingface-space-README.md` as the Space's
   `README.md` (Spaces read their settings from its header).
3. In the Space settings add the variables `CRAWL_ON_START=1` and `CRAWLER_CONTACT=<you>`, and
   `ALERTS_ENABLED=false`.

## Option 3: any machine with Docker

```bash
docker build -t hybrid-search .                 # or: docker build -f Dockerfile.lite -t hybrid-search-lite .
docker run -p 7860:7860 -e CRAWL_ON_START=1 -e CRAWLER_CONTACT=you@example.com hybrid-search
```

If you built the catalog and indexes locally first (`python -m crawler refresh` and
`scripts/build_live_index.py`), they're copied into the image and used directly, so the container
starts without crawling.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `SEARCH_MODE` | `full` (`lite` in `Dockerfile.lite`) | `lite` = BM25 + re-ranker for text, ONNX CLIP for photos |
| `PHOTO_SEARCH` | `on` | Lite mode only: `off` drops photo search and saves ~200 MB |
| `PHOTO_INDEX_MAX` | all | Lite mode only: index at most this many product photos |
| `CLIP_ONNX_URL` / `CLIP_ONNX_PATH` | Hugging Face `Xenova/clip-vit-base-patch32` (8-bit) | Where the photo model is downloaded from / kept |
| `CRAWL_ON_START` | `0` | Crawl and index at startup when the image has no catalog |
| `CRAWL_MAX_PRODUCTS` | `200` | Per-store cap for that startup crawl |
| `CRAWLER_CONTACT` | – | Your email/URL, sent in the crawler's User-Agent |
| `CATALOG` | `auto` | `live`, `demo`, or `auto` (live when its indexes exist) |
| `ALERTS_ENABLED` | `true` | Allow price-alert sign-ups |
| `LOG_IMPRESSIONS` | `true` | Log what was shown, for retraining |
| `FEEDBACK_DB` | `data/feedback.db` | Where clicks, impressions and alerts are stored |
| `ANTHROPIC_API_KEY` | – | Optional: smarter query parsing with Claude (full build) |
