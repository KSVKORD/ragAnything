#!/usr/bin/env bash
# Run on the OFFLINE server, in the folder where rag-offline-bundle.tgz was extracted.
# No internet, no git, no downloads — everything is in the bundle.
set -euo pipefail

echo ">> loading all images from rag-images.tar…"
docker load -i rag-images.tar

if [ ! -f .env ]; then
  cp .env.example .env
  echo ">> created .env — EDIT IT NOW: set DASHSCOPE_API_KEY (+ region). Then re-run this script."
  exit 0
fi

echo ">> starting the stack (no pull — all images already loaded)…"
docker compose up -d
docker compose ps
echo ">> up. Test:  curl -s localhost:8001/health"
