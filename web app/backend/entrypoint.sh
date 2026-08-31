#!/bin/bash
set -euo pipefail

# Directories
ARTIFACTS_DIR="/app/artifacts"
SYMBOLIC_DIR="/app/symbolic-moe"
SAVES_DIR="${ARTIFACTS_DIR}/saves"
PROFILES_FILE="${SYMBOLIC_DIR}/profiles.json"
SKILLS_FILE="${SYMBOLIC_DIR}/skills.txt"

# Download artifacts from GCS if not present
if [ ! -d "${SAVES_DIR}" ] || [ ! -f "${PROFILES_FILE}" ] || [ ! -f "${SKILLS_FILE}" ]; then
    echo "Downloading model artifacts from GCS..."
    mkdir -p "${ARTIFACTS_DIR}" "${SYMBOLIC_DIR}"
    
    # Use gcloud with service account credentials (mounted via Cloud Run)
    gcloud storage cp -r gs://serml-app-artifacts/artifacts/saves "${ARTIFACTS_DIR}/" 2>/dev/null || true
    gcloud storage cp gs://serml-app-artifacts/artifacts/symbolic-moe/profiles.json "${PROFILES_FILE}" 2>/dev/null || true
    gcloud storage cp gs://serml-app-artifacts/artifacts/symbolic-moe/skills.txt "${SKILLS_FILE}" 2>/dev/null || true
    
    echo "Artifacts downloaded."
fi

# Symlink for backward compatibility (code expects ../../symbolic-moe from backend)
ln -sfn "${SYMBOLIC_DIR}" /app/symbolic-moe 2>/dev/null || true
ln -sfn "${SAVES_DIR}" /app/saves 2>/dev/null || true

# Start uvicorn
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1