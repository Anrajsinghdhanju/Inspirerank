# InspireRank

A production-oriented multimodal recommendation and discovery system.

## Milestone 1

This starter gets the local platform running:

- FastAPI API
- PostgreSQL 17 + pgvector
- Redis
- Amazon Reviews 2023 streaming ingestion
- `items`, `users`, and `interactions` tables
- health and dataset-stat endpoints

We intentionally begin with **Amazon Reviews 2023 / Handmade Products** because it is visually rich and manageable for local development. The architecture is dataset-agnostic so we can later scale to Home & Kitchen and additional categories.

## 1. Prerequisites

- WSL2 / Ubuntu
- Python 3.12+
- Docker Desktop with WSL integration
- Git

## 2. Create the environment

From the repo root:

```bash
cp .env.example .env

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e "./apps/api[dev,data]"
```

## 3. Start infrastructure

```bash
docker compose up -d
docker compose ps
```

Postgres is exposed on `localhost:5432`.
Redis is exposed on `localhost:6379`.

## 4. Create application tables

Run this from the repo root:

```bash
PYTHONPATH=apps/api python -m app.db.init_db
```

## 5. Run the API

```bash
uvicorn app.main:app --reload --app-dir apps/api --host 0.0.0.0 --port 8000
```

Open:

- http://localhost:8000/
- http://localhost:8000/health
- http://localhost:8000/docs
- http://localhost:8000/api/v1/stats

## 6. Ingest a small real-data sample

In a second terminal, with the virtual environment active:

```bash
python apps/api/scripts/ingest_amazon.py \
  --max-items 2000 \
  --max-reviews 10000
```

The script streams Amazon Reviews 2023 data instead of downloading the full dataset first.

Then check:

```bash
curl http://localhost:8000/api/v1/stats
```

## Why the first dataset is Handmade Products

The end product is a visual discovery/recommendation platform, so Handmade Products gives us useful image-heavy content without forcing a huge Home & Kitchen download during development.

We will later add:

1. SigLIP image/text embeddings
2. pgvector semantic search
3. interaction event APIs
4. personalized candidate retrieval
5. two-tower recommendation
6. ranking + diversity reranking
7. real-time Redis features
8. event streaming
9. evaluation and load benchmarks

## Important

Do not put invented performance numbers in the README or resume. We will benchmark the finished system and only report measured results.
