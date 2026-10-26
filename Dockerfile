# Full engine in one container: the React site and the FastAPI search API on one URL.
# BM25 + MiniLM vectors + CLIP photo search + LightGBM re-ranking. Needs about 2 GB of memory.
#   docker build -t hybrid-search .
#   docker run -p 7860:7860 -e CRAWL_ON_START=1 -e CRAWLER_CONTACT=you@example.com hybrid-search
# For 512 MB hosts (e.g. Render's free tier) use Dockerfile.lite instead.

FROM node:20-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# Empty API URL: the site calls the API on its own origin.
ENV VITE_GATEWAY_URL=""
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.cache/huggingface
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image far smaller than the default CUDA build.
COPY search_service/requirements.txt /tmp/requirements.txt
COPY crawler/requirements.txt /tmp/crawler-requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r /tmp/requirements.txt -r /tmp/crawler-requirements.txt

# Download the models at build time so the first search after a restart isn't slow.
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2'); \
from transformers import CLIPModel as M, CLIPProcessor as P; M.from_pretrained('openai/clip-vit-base-patch32'); P.from_pretrained('openai/clip-vit-base-patch32')"

COPY . /app
COPY --from=frontend /frontend/dist /app/frontend_dist

# Hugging Face Spaces run containers as user 1000; give it the folders it writes to.
RUN chown -R 1000:1000 /app
USER 1000
ENV FRONTEND_DIST=/app/frontend_dist SEARCH_MODE=full PORT=7860 HF_HUB_OFFLINE=1
EXPOSE 7860
CMD ["sh", "deploy/start.sh"]
