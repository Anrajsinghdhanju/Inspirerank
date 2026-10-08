# InspireRank Milestone 10 — Real-Time Personalization

This milestone turns the static personalized feed into an online recommender.

## Architecture

```text
Like / Save / Hide
        |
        v
POST /api/v1/interactions
        |
        +--> PostgreSQL permanent event log
        |
        +--> Redis recent-feedback window
                    |
                    v
            live preference vector
                    |
                    v
          hybrid feed reranking
```

The neural network is NOT retrained after every click.

Offline model:
- long-term preference signal

Redis:
- short-term preference signal
- 30 latest events
- 7-day TTL

PostgreSQL:
- permanent event audit/history

## Feedback weights

```text
Like             +0.8
Save             +1.2
Not interested   -0.9
```

Recent feedback modifies the semantic user vector immediately. Positive recent
items are also appended to the learned history tower.

## 1. Commit Milestone 9

From the repo root:

```powershell
git add .
git commit -m "milestone 9: productionize personalized recommendation frontend"
```

## 2. Apply this patch

Copy the patch contents over the current repository.

## 3. Make sure Postgres + Redis are running

```powershell
docker compose up -d postgres redis
```

Check:

```powershell
docker compose ps
```

Both services should be healthy.

## 4. Initialize the permanent interaction-event table

```powershell
$env:PYTHONPATH="apps/api"
python apps/api/scripts/init_realtime.py
```

Expected:

```text
Realtime interaction table is ready.
```

## 5. Restart FastAPI

```powershell
$env:PYTHONPATH="apps/api"
uvicorn app.main:app --reload --app-dir apps/api --host 0.0.0.0 --port 8000
```

## 6. Restart Next.js

In another terminal:

```powershell
cd apps/web
npm run dev
```

## 7. Test

Open:

```text
http://localhost:3000
```

Every recommendation now has:

- Like
- Save
- Hide

After clicking one:

1. the event is written to PostgreSQL,
2. recent feedback is written to Redis,
3. the feed reloads,
4. the clicked item disappears,
5. the live-signal counter increases,
6. recommendations are reranked using the recent preference vector.

## Debug endpoints

Recent Redis feedback:

```text
GET /api/v1/interactions/{user_id}/recent
```

Reset only live Redis state:

```text
DELETE /api/v1/interactions/{user_id}/recent
```

The PostgreSQL audit log is deliberately preserved when live state is reset.

## Useful database check

Inside Postgres:

```sql
SELECT
    user_id,
    item_id,
    event_type,
    created_at
FROM realtime_interactions
ORDER BY created_at DESC
LIMIT 20;
```

## Success test

Pick a demo user with a clear theme.

1. Save/like several items of another visual theme.
2. Observe the Live signals count rise.
3. Confirm the clicked cards disappear.
4. Confirm subsequent recommendations begin shifting toward the new theme.
5. Click Hide on an unwanted item and verify it does not immediately return.

## Next milestone

After this works:

- event analytics endpoint
- semantic search in the frontend
- candidate retrieval -> ranking -> diversity reranking
- instrumentation / latency benchmarks
