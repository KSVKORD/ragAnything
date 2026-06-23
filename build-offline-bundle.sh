#!/usr/bin/env bash
# Build ONE self-contained offline bundle: all 5 images + compose + env.
# Run on a machine that already has the images built/pulled (e.g. the current server).
# The target server needs NO internet and NO git — just this one file.
set -euo pipefail

REGISTRY="${REGISTRY:-docker.m.daocloud.io}"
OUT="${1:-rag-offline-bundle.tgz}"

echo ">> saving all 5 images (this is the big step)…"
docker save -o rag-images.tar \
  raganything-app:latest \
  raganything-dotsocr:latest \
  "$REGISTRY/library/postgres:17" \
  "$REGISTRY/qdrant/qdrant:v1.18.1" \
  "$REGISTRY/library/neo4j:5.26"

echo ">> packing images + compose + env into $OUT …"
tar czf "$OUT" docker-compose.yml .env.example README.md rag-images.tar
rm -f rag-images.tar

echo ">> done: $OUT ($(du -h "$OUT" | cut -f1))"
echo "   Transfer it to the offline server, then run install-offline.sh there."
