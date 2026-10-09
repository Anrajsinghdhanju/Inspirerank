# Milestone 16 — Public Deployment + Recruiter Polish

## Adds

- Caddy reverse proxy + automatic HTTPS
- single-origin deployment
- CPU-only PyTorch deployment image
- frontend health check
- GitHub Actions CI
- VM deployment script
- recruiter-facing README
- interview story
- resume bullets

## 1. Commit Milestone 15

Check:

```powershell
git status
```

Make sure `.env.production` is ignored, then:

```powershell
git add .
git commit -m "milestone 15: productionize services with caching health checks and observability"
```

## 2. Apply this patch

Also add to `.gitignore`:

```text
.env.production
.env.deploy
```

## 3. Local checks

```powershell
pytest apps/api/tests/test_catalog_quality.py -q
ruff check apps/api/app apps/api/tests
```

Frontend:

```powershell
cd apps/web
npm audit
npm run build
cd ../..
```

## 4. Optional deployment-image size test

Only when Docker Desktop is healthy:

```powershell
docker build -f apps/api/Dockerfile.deploy -t inspirerank-api-deploy .
docker image ls inspirerank-api-deploy
```

The deployment image installs CPU-only PyTorch first to avoid unnecessary CUDA
libraries.

## 5. Public deployment

Follow `docs/DEPLOY_PUBLIC.md`.

## 6. After the public URL works

Replace the README's public-demo placeholder with the real HTTPS URL, then:

```powershell
git add .
git commit -m "milestone 16: add public deployment and recruiter-facing documentation"
git push
```

Do not claim public deployment on the resume until the URL is actually live.
