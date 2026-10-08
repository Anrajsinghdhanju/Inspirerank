# InspireRank Milestone 13 — Query-Conditioned User Interests

## Why

Milestone 12 reduced broad-query overlap from ~97.5% to ~85.8%, but the same
few top results still appeared for most users.

The reason: averaging a user's full history produces an overly generic
long-term taste vector.

A user with hundreds of interactions may have multiple distinct interests:

```text
jewelry
painting
origami
sewing
resin
scrapbooking
```

A single average collapses these into a generic "craft" representation.

## New search user tower

For each search query:

```text
user's history
      |
compute similarity to query
      |
top 24 relevant historical items
      |
query-attention + recency weighting
      |
query-conditioned user profile
```

So:

```text
query = "craft supplies"
```

might activate:

```text
User A -> jewelry / resin history
User B -> quilting / sewing history
User C -> paper / origami history
User D -> painting history
```

rather than giving all four users a generic craft vector.

## Search scoring

Broad query approximation:

```text
query relevance              48%
query-conditioned history    30%
global taste                  8%
live Redis preference         9%
learned behavior              5%
```

Specific queries remain query-dominant.

## Recency

Within the query-relevant history, newer interactions receive a mild
exponential recency boost with a 35-interaction half-life.

## Apply

Commit Milestone 12:

```powershell
git add .
git commit -m "milestone 12: add two-stage adaptive personalized reranking"
```

Copy this patch over the repository.

Restart FastAPI. The frontend requires no changes for this experiment.

## Re-run the broad-query test

```powershell
$env:PYTHONPATH="apps/api"

python apps/api/scripts/compare_search_personalization.py --query "craft supplies"
```

The script now also prints the historical items receiving the most attention
for each user.

Then:

```powershell
python apps/api/scripts/compare_search_personalization.py --query "creative gift"
```

And verify a specific query remains stable:

```powershell
python apps/api/scripts/compare_search_personalization.py --query "origami paper with Japanese patterns"
```

## What to look for

Broad query:
- users should activate different historical interests
- top-five results should visibly differ
- pairwise overlap should fall from the Milestone 12 result

Specific query:
- overlap should remain high
- explicit query intent should still dominate

Do not target zero overlap. Personalized search should remain relevant, not
random.
