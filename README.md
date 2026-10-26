# 🧵 Dhaaga: hybrid search across India's fashion brands

[![CI](https://github.com/kwazzy-coder/Hybrid-Product-Search-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/kwazzy-coder/Hybrid-Product-Search-Engine/actions/workflows/ci.yml)

*Dhaaga* (धागा) is Hindi for thread: one thread through many stores. It's a multimodal fashion search engine over **real products crawled from Indian fashion brands**, combining **Lexical Keyword Search (BM25)**, **Dense Text Vector Search (FAISS + MiniLM-L6-v2)**, and **Text-to-Image and Photo Search (CLIP ViT-B/32)**. Results are fused using **Reciprocal Rank Fusion (RRF)**, re-ranked via a **12-feature LightGBM LambdaRank model**, and filtered by category, price and **size in stock**.

Includes **click logging with serve-time feature logging**, **LTR retraining on real clicks**, an **offline evaluation harness on the live catalog** (LLM-assisted relevance labels), **price history and price-drop alerts**, CI, and a one-container deployment.

---

## 🌟 Key Features

* **⚡ 3-Way Parallel Hybrid Retrieval**: Queries execute across BM25, FAISS dense text, and CLIP text-to-image indexes simultaneously using `asyncio.gather`, merged via scale-invariant Reciprocal Rank Fusion ($k=60$).
* **🎯 12-Feature Learning-to-Rank (LTR)**: LightGBM LambdaRank model (`objective="lambdarank"`, `metric="ndcg"`) optimized for top-K slot precision based on category match, color binding, ratings, discounts, CLIP visual similarity, and cross-platform price competitiveness.
* **🖼️ Search by Photo**: Upload or drop a photo in the search bar to find visually similar products (CLIP image embeddings), optionally narrowed with text like *"under 1500"*.
* **📏 Size in Stock**: Only show items available in your size, from a size chip or the query itself (*"black shirt size L"*, *"jeans 32 waist"*). Sizes are normalized across stores (XXL/2XL, UK 8/8, Free Size).
* **🕸️ Live Catalog from Real Fashion Stores**: A polite, robots.txt-aware crawler (`crawler/`) collects real products from Indian fashion brands (Libas, Suta, Powerlook, The Bear House, Bonkers Corner), with real prices, MRP, sizes, stock and product links, and tracks price history for price-drop badges. See [`crawler/README.md`](crawler/README.md).
* **🛍️ Cross-Store Matching**: In live mode, listings of the same physical product in different stores are matched by CLIP image similarity (≥0.92), confirmed by brand/title similarity, category and price sanity checks, to show where it's cheapest. The demo catalog simulates this with synthetic duplicates across *Amazon, Myntra, Ajio, Flipkart* labels.
* **🔄 Learning from Real Clicks**: Every search logs what was shown together with the exact 12 features each result had at that moment; clicks are logged with their position. `train_ltr_live.py` retrains on these with skip-above negatives (to reduce position bias) and only ships a model that beats the current one on held-out searches.
* **📊 Evaluation on the Live Catalog**: `eval_live.py` pools candidates from every pipeline stage, labels them 0–3 with Claude (spot-checked by hand), and reports NDCG@5/10, Recall@20, MRR and latency per stage, so each component's contribution is measured.
* **🔔 Price History & Alerts**: Daily crawls record price changes; cards show a price-history line, "lowest in 30 days" and price-drop badges, and shoppers can ask to be emailed when a product drops below their price.
* **🧪 CI & Deployment**: GitHub Actions runs the test suite and frontend lint/build on every push; one container serves the site and API (lite build fits a 512 MB free tier, see [`DEPLOY.md`](DEPLOY.md)).

---

## 🏗️ Architecture & Pipeline Data Flow

```
                               ┌────────────────────────────────┐
                               │  User Query / Image Upload     │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │       Query Understanding      │
                               │   (Intent, Category, Color,    │
                               │  Color Bindings, Synonyms)     │
                               └───────────────┬────────────────┘
                                               │
             ┌─────────────────────────────────┼─────────────────────────────────┐
             │ (Parallel asyncio.gather)       │                                 │
             ▼                                 ▼                                 ▼
   ┌───────────────────┐             ┌───────────────────┐             ┌───────────────────┐
   │   BM25 Lexical    │             │   FAISS Vector    │             │ CLIP Text-to-Image│
   │  (rank-bm25 index)│             │ (MiniLM-L6-v2)    │             │   (ViT-B/32)      │
   └─────────┬─────────┘             └─────────┬─────────┘             └─────────┬─────────┘
             │                                 │                                 │
             └─────────────────────────────────┼─────────────────────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │   3-Way Weighted Fusion (RRF)  │
                               │     Weights: [1.0, 1.0, 0.8]   │
                               └───────────────┬────────────────┘
                                               │ (Top-100 Candidates)
                                               ▼
                               ┌────────────────────────────────┐
                               │    Candidate Hydration         │
                               │  (MongoDB / Local Catalog JSON)│
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │   12-Feature LTR Re-Ranker     │
                               │   (LightGBM LambdaRank Model)  │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │   Source Diversity Constraint  │
                               │   (MMR-like Greedy Penalty)    │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │   Paginated API Response +     │
                               │   Cross-Platform Price Compare │
                               └────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Layer | Technologies & Frameworks |
|---|---|
| **Backend API** | Python 3.12, FastAPI, Pydantic v2, Starlette, Uvicorn (ASGI) |
| **Lexical Search** | `rank-bm25` (BM25Okapi token scoring) |
| **Vector Search** | FAISS (`IndexIDMap`, `IndexFlatIP`), Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **Visual Search** | HuggingFace Transformers, PyTorch (`openai/clip-vit-base-patch32`, 512-d) |
| **Re-ranking & ML** | LightGBM LambdaRank (`objective="lambdarank"`), NumPy, Scikit-learn |
| **Databases** | MongoDB (PyMongo), SQLite3 (`click_events`), Redis (query cache fallback) |
| **Frontend** | React 18, Vite, CSS3 Glassmorphism UI, JavaScript Fetch API |
| **Containerization** | Docker, Docker Compose |

---

## 🚀 Quick Start Guide

### Prerequisites
* **Python**: `3.11+` (Python 3.12 recommended)
* **Node.js**: `v18+` (for React frontend)

---

### 1. Installation & Environment Setup

```bash
# Clone repository
git clone https://github.com/kwazzy-coder/Hybrid-Product-Search-Engine.git
cd Hybrid-Product-Search-Engine

# Install Python backend dependencies
pip install -r search_service/requirements.txt

# Install React frontend dependencies
cd frontend
npm install
cd ..
```

---

### 2. Dataset & Vector Index Building

**Option A: live catalog from real fashion stores (recommended)**

```bash
pip install -r crawler/requirements.txt
python -m crawler refresh                    # crawl stores, export catalog, download thumbnails
cd search_service
python scripts/build_live_index.py           # BM25 + MiniLM + CLIP indexes + cross-store matching
cd ..
```

Set your contact in `crawler/sites.toml` first. Details, scheduling and the list of stores are in [`crawler/README.md`](crawler/README.md).
The API uses the live catalog automatically once its indexes exist (`CATALOG=demo` switches back).

**Option B: demo catalog** (2,300 HuggingFace fashion items with synthetic prices, ratings and store labels):

```bash
cd search_service
python scripts/build_dataset.py
cd ..
```

---

### 3. Launch Services

#### **Start FastAPI Backend (Port 8000)**:
```bash
cd search_service
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
*API Interactive Swagger Docs*: `http://localhost:8000/docs`

#### **Start React Frontend (Port 5173)**:
```bash
cd frontend
npm run dev
```
*Frontend UI*: `http://localhost:5173`

---

## 🔄 Learning from Clicks (LTR Retraining)

The website logs every search's results with their ranking features, and every click with its position. Once you've collected 30+ searches with clicks:

```bash
cd search_service
python scripts/train_ltr_live.py
```

It trains on real behaviour (clicked = positive; shown above a click but skipped = negative; results below the last click are dropped as probably unseen), compares against the current model on the newest 20% of searches, and saves to `models_live/ltr_model.lgb` only if it's better. Restart the API or `POST /admin/reload_ltr` to use it.

<details><summary>Older synthetic pipeline (demo catalog)</summary>

`simulate_clicks.py` → `build_hard_negatives.py` → `retrain_ltr.py --mode hybrid`. These train on simulated clicks, so they demonstrate the mechanics rather than learn real preferences.
</details>

---

## 📊 Evaluation

Measure search quality on the live catalog, stage by stage:

```bash
cd search_service
python scripts/eval_live.py pool      # candidates for the 40 queries in data/eval/queries.txt
python scripts/eval_live.py judge     # label 0-3 with Claude (needs ANTHROPIC_API_KEY), or label by hand in data/eval/labels.csv
python scripts/eval_live.py run       # NDCG@5/10, Recall@20, MRR, latency -> data/eval/results.md
```

Spot-check the LLM labels in `data/eval/labels.csv` before trusting the numbers, then paste `data/eval/results.md` here.

<details><summary>Earlier benchmark on the synthetic demo catalog</summary>

`python search_service/scripts/evaluate.py` compares re-ranking off vs on for the demo catalog. Its relevance labels come from the same category/colour rules used to simulate the training clicks, so treat it as a sanity check rather than a quality measure. Mean NDCG@5 went from 0.890 (RRF) to 0.905 (LTR).

```
========================================================================================
QUERY                    | RRF (OFF) N@5 | LTR (ON) N@5  | N@5 LIFT   | LTR (ON) N@10
----------------------------------------------------------------------------------------
red dress                | 0.1429        | 0.4286        | +200.0%    | 0.6038       
blue shirt               | 0.8062        | 1.0000        | +24.0%     | 1.0000       
ethnic kurta             | 0.4469        | 0.8539        | +91.1%     | 0.6840       
sandals                  | 0.8539        | 1.0000        | +17.1%     | 0.9266       
office wear              | 1.0000        | 1.0000        | +0.0%      | 1.0000       
black formal shoes       | 1.0000        | 1.0000        | +0.0%      | 1.0000       
========================================================================================
MEAN AGGREGATE METRICS   | 0.8900        | 0.9052        | +1.7%      | 0.8963       
========================================================================================
```
</details>

---

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

The crawler is tested against local fake stores (pagination, robots.txt blocks, 429 back-off, 403 refusals, price history), and the API end to end with FastAPI's test client (search, size filter, click and impression logging, alerts, photo search, lite mode).

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/search?q={query}` | 3-way hybrid search + LTR re-ranking + source diversity. Filters: `category`, `min_price`, `max_price`, `min_rating`, `in_stock_only`, `size` |
| `POST`| `/api/search/image` | Photo search (`file` image + optional `q` text), same filters |
| `POST`| `/api/feedback` | Logs a click `{query, query_id, clicked_id, position}` (impressions are logged automatically) |
| `POST`| `/api/alerts` | Price-drop alert `{product_id, target_price, email}`, checked after each crawl |
| `GET` | `/api/stats` | Returns index sizes, vector dimensions, and cross-platform statistics |
| `GET` | `/api/suggestions?q={q}`| Autocomplete search suggestions |
| `GET` | `/docs` | Interactive Swagger API Documentation |

---

## 🐳 Deployment

One container serves the website and the API. `Dockerfile` is the full engine (~2 GB memory); `Dockerfile.lite` is BM25 + re-ranking and fits a 512 MB free tier. `render.yaml` deploys the lite build to Render for free. See [`DEPLOY.md`](DEPLOY.md), including what to check before publishing store data.

The original multi-service setup (MongoDB, Redis, gateway) still runs with `docker-compose up --build`.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
