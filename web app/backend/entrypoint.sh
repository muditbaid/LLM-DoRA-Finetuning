#!/bin/bash
set -euo pipefail

echo "=== SERML Entrypoint Starting ==="
echo "PORT: ${PORT:-8000}"
echo "MODEL_BACKEND: ${MODEL_BACKEND:-local_inprocess}"

# Artifacts are pre-baked in image
echo "=== Artifacts pre-baked in image ==="
echo "=== Starting uvicorn on port ${PORT:-8000} ==="

exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
