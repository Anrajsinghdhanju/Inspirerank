# Public Deployment — Single Ubuntu VM

A single Linux VM is the simplest reliable public demo for the current
architecture because it preserves Docker, PostgreSQL, Redis, local model
artifacts, and persistent volumes.

## Suggested starting shape

```text
4 vCPU
8 GB RAM
40+ GB SSD
Ubuntu 24.04 LTS
```

The model-loaded API is memory-heavy, so avoid a 1–2 GB instance.

## 1. DNS

Create an `A` record:

```text
inspirerank.yourdomain.com -> SERVER_PUBLIC_IP
```

## 2. Firewall

Allow inbound:

```text
22/tcp
80/tcp
443/tcp
443/udp
```

Do not expose PostgreSQL, Redis, FastAPI, or Next.js directly.

## 3. Install Docker

Install Docker Engine and the Docker Compose plugin, then verify:

```bash
docker version
docker compose version
```

## 4. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd inspirerank_starter
```

If `data/` and `artifacts/` are excluded from Git, transfer them separately.

Expected paths:

```text
data/recsys/arts_crafts_5core/
artifacts/catalog_siglip/
artifacts/content_two_tower_v1/
```

## 5. Environment

```bash
cp .env.deploy.example .env.deploy
nano .env.deploy
```

Set:

```text
APP_DOMAIN=inspirerank.yourdomain.com
POSTGRES_PASSWORD=<strong random secret>
```

Never commit `.env.deploy`.

## 6. Deploy

```bash
chmod +x deploy/deploy.sh
./deploy/deploy.sh
```

Caddy obtains TLS automatically once DNS resolves and ports 80/443 are
reachable.

## 7. Verify

```bash
docker compose --env-file .env.deploy -f docker-compose.deploy.yml ps
```

Public endpoints:

```text
https://YOUR_DOMAIN/
https://YOUR_DOMAIN/health/ready
https://YOUR_DOMAIN/docs
https://YOUR_DOMAIN/metrics
```

## Before recruiter sharing

- verify no secrets exist in Git history;
- verify mobile layout;
- verify all product images load over HTTPS;
- add the public URL to the README;
- consider protecting `/metrics` if the demo receives meaningful public traffic.
