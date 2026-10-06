# InspireRank — Milestone 2

## Goal

Generate SigLIP image/text embeddings for real Amazon items and search them with pgvector cosine similarity.

## Apply this patch

Copy the `apps` directory from this patch into the existing repository and allow it to replace matching files.

Your existing PostgreSQL volume and ingested data are not touched.

## Install ML dependencies

From the project root with the virtual environment activated:

```powershell
python -m pip install -e "./apps/api[dev,data,ml]"
```

## Add the embeddings table + HNSW index

```powershell
python -m app.db.init_db
```

This is non-destructive. Existing `users`, `items`, and `interactions` stay in place.

## Check GPU

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Smoke test with 25 items

```powershell
python apps/api/scripts/generate_embeddings.py --limit 25 --batch-size 4
```

The first run downloads the SigLIP model weights.

## Verify

Start or restart FastAPI:

```powershell
uvicorn app.main:app --reload --app-dir apps/api --host 0.0.0.0 --port 8000
```

Visit:

- http://localhost:8000/api/v1/stats
- http://localhost:8000/docs

Search:

```text
http://localhost:8000/api/v1/search?q=wooden%20home%20decor&limit=10
```

Compare modes:

```text
/api/v1/search?q=wooden%20home%20decor&mode=image
/api/v1/search?q=wooden%20home%20decor&mode=text
/api/v1/search?q=wooden%20home%20decor&mode=multimodal
```

## Scale after the smoke test

```powershell
python apps/api/scripts/generate_embeddings.py --limit 500 --batch-size 8
```

Then later:

```powershell
python apps/api/scripts/generate_embeddings.py --limit 2000 --batch-size 8
```

Already-embedded items are skipped by default.
