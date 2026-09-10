# Build the browser UI separately so the final CUDA image contains only the
# compiled static assets, not Node.js or npm.
FROM node:24-bookworm-slim AS frontend-builder

WORKDIR /frontend
COPY ["web app/frontend/package.json", "web app/frontend/package-lock.json", "./"]
RUN npm ci
COPY ["web app/frontend", "./"]
ENV VITE_ENABLE_MOCK_FALLBACK=false \
    VITE_REDIRECT_TO_API_ORIGIN=false
RUN npm run build

# Cloud Run GPU runtime. The development image is required because Triton JIT
# compiles a small CUDA/Python helper on the first inference.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HF_HOME=/app/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/app/cache/huggingface/hub \
    TRITON_CACHE_DIR=/app/cache/triton

# Ubuntu 22.04's python3-pip uses Python 3.10. Install the matching development
# headers so Triton's cuda_utils.c can include Python.h at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin PyTorch to the CUDA 12.4 wheel lane, then install the application stack.
COPY ["web app/backend/requirements.txt", "/tmp/requirements.txt"]
RUN python3 -m pip install --no-cache-dir \
      --index-url "${PYTORCH_INDEX_URL}" \
      torch==2.6.0 && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    python3 -m pip check && \
    python3 -c "import os, sysconfig; p=os.path.join(sysconfig.get_path('include'), 'Python.h'); print(p); assert os.path.isfile(p), p" && \
    python3 -c "import bitsandbytes, peft, torch, transformers, triton; print({'torch': torch.__version__, 'cuda': torch.version.cuda, 'triton': triton.__version__, 'bitsandbytes': bitsandbytes.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__})"

# Create the unprivileged user before copying the large model so ownership is
# recorded directly in the copy layer instead of duplicating it with chown.
RUN useradd --create-home --uid 1000 appuser && \
    install -d -o appuser -g appuser /app/symbolic-moe /app/artifacts /app/cache

# Canonical runtime source and router inputs come from the repository root.
COPY --chown=appuser:appuser symbolic-moe/*.py /app/symbolic-moe/
COPY --chown=appuser:appuser symbolic-moe/profile_pool_skills.jsonl /app/symbolic-moe/
COPY --chown=appuser:appuser symbolic-moe/profiles.json /app/symbolic-moe/
COPY --chown=appuser:appuser symbolic-moe/skills.txt /app/symbolic-moe/

# cloudbuild.yaml stages private GCS artifacts here before docker build runs.
COPY --chown=appuser:appuser ["web app/backend/model-artifacts", "/app/artifacts"]
RUN test -d /app/artifacts/saves && \
    test -f /app/artifacts/symbolic-moe/profiles.json && \
    test -f /app/artifacts/symbolic-moe/skills.txt && \
    test -f /app/artifacts/base-model/config.json && \
    test -f /app/artifacts/base-model/tokenizer_config.json && \
    test -f /app/artifacts/base-model/.complete && \
    test -f /app/artifacts/base-model/manifest.json && \
    { test -f /app/artifacts/base-model/model.safetensors.index.json || \
      test -f /app/artifacts/base-model/model.safetensors; } && \
    for adapter in dynahate tweeteval_offensive kaggle_cyberbullying jigsaw_threat; do \
      test -f "/app/artifacts/saves/llama31-8b/${adapter}/qlora/adapter_config.json"; \
      test -f "/app/artifacts/saves/llama31-8b/${adapter}/qlora/adapter_model.safetensors"; \
    done && \
    cp /app/artifacts/symbolic-moe/profiles.json /app/symbolic-moe/profiles.json && \
    cp /app/artifacts/symbolic-moe/skills.txt /app/symbolic-moe/skills.txt && \
    ln -s /app/artifacts/saves /app/saves

COPY --chown=appuser:appuser ["web app/backend/app", "/app/app"]
COPY --from=frontend-builder --chown=appuser:appuser /frontend/dist /app/static
COPY --chown=appuser:appuser ["web app/backend/entrypoint.sh", "/entrypoint.sh"]
RUN test -f /app/static/index.html && \
    sed -i 's/\r$//' /entrypoint.sh && \
    chmod +x /entrypoint.sh

USER appuser

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
