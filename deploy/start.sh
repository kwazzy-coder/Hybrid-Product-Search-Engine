#!/bin/sh
# Container entry point: make sure there's a catalog and indexes, then start the API
# (which also serves the built website from FRONTEND_DIST).
set -e
cd "$(dirname "$0")/.."
MODE="${SEARCH_MODE:-full}"

# No catalog baked into the image (e.g. built from the public repo): crawl a small one now.
if [ "${CRAWL_ON_START:-0}" = "1" ] && [ ! -f data/products_live.json ]; then
  echo "No catalog found: crawling up to ${CRAWL_MAX_PRODUCTS:-200} products per store ..."
  python -m crawler crawl --max-products "${CRAWL_MAX_PRODUCTS:-200}" || echo "Crawl had errors; continuing with what was collected"
  python -m crawler export || true
  if [ "$MODE" != "lite" ]; then
    python -m crawler images || true
  fi
fi

if [ -f data/products_live.json ] && [ ! -f search_service/models_live/bm25.pkl ]; then
  echo "Building search indexes ($MODE mode) ..."
  if [ "$MODE" = "lite" ]; then
    (cd search_service && python scripts/build_live_index.py --lite)
  else
    (cd search_service && python scripts/build_live_index.py)
  fi
fi

cd search_service
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
