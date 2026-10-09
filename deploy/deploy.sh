#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env.deploy ]]; then
  echo "Missing .env.deploy"
  echo "Copy .env.deploy.example to .env.deploy and edit it first."
  exit 1
fi

git pull --ff-only

docker compose \
  --env-file .env.deploy \
  -f docker-compose.deploy.yml \
  up -d --build --remove-orphans

sleep 10

docker compose \
  --env-file .env.deploy \
  -f docker-compose.deploy.yml \
  ps

APP_DOMAIN="$(grep '^APP_DOMAIN=' .env.deploy | cut -d= -f2-)"

curl --fail --retry 20 --retry-delay 5 \
  "https://${APP_DOMAIN}/health/ready"

echo
echo "Deployment healthy: https://${APP_DOMAIN}"
