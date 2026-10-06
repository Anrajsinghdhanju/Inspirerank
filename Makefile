.PHONY: infra-up infra-down api install init-db ingest-smoke stats

infra-up:
	docker compose up -d

infra-down:
	docker compose down

install:
	python -m pip install -e "./apps/api[dev,data]"

init-db:
	python -m app.db.init_db

api:
	uvicorn app.main:app --reload --app-dir apps/api --host 0.0.0.0 --port 8000

ingest-smoke:
	python apps/api/scripts/ingest_amazon.py --max-items 2000 --max-reviews 10000

stats:
	curl -s http://localhost:8000/api/v1/stats
