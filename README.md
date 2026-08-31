# Skill-Based LLM Driven Expert Routing System For Multilabel Harmful Speech Detection

Product-facing branch for a multilabel harmful-speech detection system that combines symbolic skill inference, a Bernoulli Naive Bayes expert router, LoRA expert adapters, and a web-serving layer.

[Web UI](https://sberhsd.muditb0712.workers.dev)  
[Project Page](https://muditbaid.github.io/Skill-Based-Expert-Routing-System-For-Multilabel-Hate-Speech-Detection/)  
[API (Cloud Run)](https://serml-api-XXXXXX-uc.a.run.app/docs)

## Project Preview

The repository-hosted presentation page is previewed below. Click the image to open the full page.

<p>
  <a href="https://muditbaid.github.io/Skill-Based-Expert-Routing-System-For-Multilabel-Hate-Speech-Detection/">
    <img src="docs/assets/project-preview-slide-1.png" alt="Project page preview slide 1" width="100%" />
  </a>
</p>

See the complete deck here: [Open the full project page](https://muditbaid.github.io/Skill-Based-Expert-Routing-System-For-Multilabel-Hate-Speech-Detection/)

## Overview

This repository packages the deployment-ready slice of the project:

- a symbolic skill extractor built on Llama 3.1
- a Bernoulli NB router that selects the top-2 experts from inferred skills
- four task specialists for hate, offense, bullying, and threat detection
- a shared-runtime adapter-switching backend that loads the base model once
- a FastAPI service that exposes the pipeline behind a web UI

The branch is intentionally trimmed to the core runtime, artifacts, and web app assets needed to demonstrate and serve the system.

## Project Page

This branch is GitHub Pages-ready.

- Source file: `docs/index.html`
- Preview image: `docs/assets/project-preview-slide-1.png`
- Expected Pages URL: `https://muditbaid.github.io/Skill-Based-Expert-Routing-System-For-Multilabel-Hate-Speech-Detection/`

To enable it on GitHub:

1. Open repository settings.
2. Go to `Pages`.
3. Set the source to `Deploy from a branch`.
4. Select `main` and folder `/docs`.

After that, the project presentation page will render as a live site directly from the repository.

## System Architecture

### 1. Skill inference

The pipeline first predicts ontology-level skills from raw text using `symbolic-moe/skill_inference.py`. Each post is sampled multiple times, parsed into a controlled vocabulary from `symbolic-moe/skills.txt`, and consolidated through vote thresholds.

### 2. Expert routing

The predicted skills are passed to the Bernoulli Naive Bayes router in `symbolic-moe/route_and_predict_nb.py`. The router is trained from `symbolic-moe/profile_pool_skills.jsonl` and selects the top-2 experts for each example.

### 3. Expert execution

Each selected expert is a LoRA adapter mounted on a shared Llama 3.1 base model. `symbolic-moe/model_utils.py` keeps the base model resident and switches adapters instead of reloading a full checkpoint per task.

### 4. Final aggregation

`symbolic-moe/predict_post.py` runs the full single-post flow:

1. infer skills
2. route top-2 experts
3. execute only the selected adapters
4. keep label matches for the target task
5. return multilabel output with confidences

### 5. Serving layer

`web app/backend/` wraps the pipeline in FastAPI and exposes:

- `POST /api/detect`
- `GET /health/live`
- `GET /health/ready`

The backend can run locally in-process or behind a Colab/remote GPU workflow.

## LangSmith Observability

LangSmith is used for pipeline observability and evaluation workflows rather than the live request path.

Current integrations:

- `symbolic-moe/skill_inference.py` can trace skill-prediction runs
- `symbolic-moe/route_and_predict_nb.py` can trace router scoring, selected experts, expert inference, and evaluation outcomes
- `symbolic-moe/evaluate_outputs.py` can log evaluation traces
- `symbolic-moe/langsmith_workflows.py` supports:
  - dataset upload
  - trace backfill from routed outputs
  - review queue creation
  - hard-case summarization

This makes it possible to inspect difficult examples, compare routing behavior across evaluation splits, and curate annotation/review queues for failure analysis.

Environment flags:

```bash
export LANGSMITH_API_KEY=...
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=symbolic-moe
```

See `symbolic-moe/commands.md` for concrete tracing and review commands.

## Repository Layout

```text
.
├── docs/
│   ├── assets/
│   │   ├── project-preview-slide-1.png
│   │   └── project-preview-slide-2.png
│   └── index.html
├── README.md
├── requirements.txt
├── saves/
│   └── llama31-8b/
├── symbolic-moe/
│   ├── build_profiles.py
│   ├── config.py
│   ├── evaluate_outputs.py
│   ├── io_utils.py
│   ├── langsmith_utils.py
│   ├── langsmith_workflows.py
│   ├── model_utils.py
│   ├── predict_post.py
│   ├── route_and_predict_nb.py
│   ├── skill_inference.py
│   ├── skill_parsing.py
│   ├── profile_pool.jsonl
│   ├── profile_pool_skills.jsonl
│   ├── validation_pool.jsonl
│   ├── validation_pool_skills.jsonl
│   ├── test_pool.jsonl
│   ├── test_pool_skills.jsonl
│   ├── profiles.json
│   └── skills.txt
└── web app/
    ├── backend/
    ├── COLAB_BACKEND.md
    └── SERML_Backend.ipynb
```

## Core Artifacts

The branch keeps the main runtime artifacts required by the deployed system:

- `symbolic-moe/skills.txt`: skill vocabulary used by the router
- `symbolic-moe/profile_pool.jsonl`: profile-construction pool
- `symbolic-moe/profile_pool_skills.jsonl`: skill-annotated router training pool
- `symbolic-moe/validation_pool.jsonl`: validation pool
- `symbolic-moe/test_pool.jsonl`: test pool
- `symbolic-moe/profiles.json`: learned expert profiles
- `saves/llama31-8b/*/qlora`: saved expert adapters

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Run the backend

```bash
cd "web app/backend"
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Local API example

```bash
curl -X POST "http://127.0.0.1:8000/api/detect" \
  -H "Content-Type: application/json" \
  -d '{"text":"I will find you and hurt you."}'
```

## Rebuilding the Pipeline

If profiles or pool artifacts need to be regenerated, the intended order is:

1. run `symbolic-moe/skill_inference.py` on the desired pool
2. rebuild profiles with `symbolic-moe/build_profiles.py`
3. evaluate routing with `symbolic-moe/route_and_predict_nb.py`
4. score outputs using `symbolic-moe/evaluate_outputs.py`

For single-post inference and deployment, `symbolic-moe/predict_post.py` is the canonical entrypoint used by the backend adapter.

## Web App

The Cloudflare-hosted frontend is available here:

[Web UI](https://sberhsd.muditb0712.workers.dev)

The repository-hosted presentation page is available here once GitHub Pages is enabled:

[Project Page](https://muditbaid.github.io/Skill-Based-Expert-Routing-System-For-Multilabel-Hate-Speech-Detection/)

For Colab-based backend serving, see:

- `web app/COLAB_BACKEND.md`
- `web app/SERML_Backend.ipynb`

## License

This repository inherits the upstream project license in `LICENSE`.

## Production Deployment

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLOUDFLARE PAGES                         │
│  https://sberhsd.muditb0712.workers.dev                         │
│  ─────────────────────────────────────────────────────────────  │
│  React + Vite + Tailwind  │  Calls /api/detect via fetch       │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTPS
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GOOGLE CLOUD RUN (GPU)                      │
│  https://serml-api-XXXXXX-uc.a.run.app                          │
│  ─────────────────────────────────────────────────────────────  │
│  FastAPI + 4 QLoRA Experts (LLaMA-3.1-8B) on L4 GPU            │
│  Startup: downloads artifacts from GCS → warmup → serve        │
│  Scales to 0 when idle (cost ≈ $0)                              │
│  Auth: Cloud Run IAP (Google OAuth)                             │
│  Rate limiting: 10 req/min per IP                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        GCS BUCKET                               │
│  gs://serml-app-artifacts/artifacts/                            │
│  ├── saves/llama31-8b/*/qlora/  (4 adapters ~8GB)              │
│  ├── symbolic-moe/profiles.json                                │
│  └── symbolic-moe/skills.txt                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Prerequisites

- GCP project `serml-app` with billing enabled
- $300 free credits (covers ~10 months of typical usage)
- Cloudflare account for Pages hosting
- HF token for Llama model access

### One-time GCP Setup

```bash
# 1. Create service account
gcloud iam service-accounts create serml-api \
  --display-name="SERML API Backend" \
  --project=serml-app

# 2. Grant GCS access
gcloud projects add-iam-policy-binding serml-app \
  --member="serviceAccount:serml-api@serml-app.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"

# 3. Create HF token secret
echo -n "your_hf_token" | gcloud secrets create hf-token --data-file=- --project=serml-app

# 4. Grant secret access
gcloud secrets add-iam-policy-binding hf-token \
  --member="serviceAccount:serml-api@serml-app.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" \
  --project=serml-app

# 5. Enable IAP
gcloud services enable iap.googleapis.com --project=serml-app

# 6. Create Artifact Registry repo
gcloud artifacts repositories create serml-repo \
  --repository-format=docker \
  --location=us-west1 \
  --project=serml-app
```

### Manual Deploy (Backend)

```bash
# Build & push
gcloud builds submit --tag us-west1-docker.pkg.dev/serml-app/serml-repo/serml-api:latest web app/backend

# Deploy to Cloud Run with L4 GPU
gcloud run deploy serml-api \
  --image=us-west1-docker.pkg.dev/serml-app/serml-repo/serml-api:latest \
  --region=us-west1 \
  --gpu=1 --gpu-type=nvidia-l4 \
  --memory=16Gi --cpu=4 \
  --min-instances=0 --max-instances=3 \
  --concurrency=1 --timeout=300 --port=8000 \
  --set-env-vars=MODEL_BACKEND=local_inprocess,SYMBOLIC_MOE_DIR=/app/symbolic-moe,APP_ENV=prod,ALLOW_ORIGINS=https://sberhsd.muditb0712.workers.dev \
  --set-secrets=HF_TOKEN=hf-token:latest \
  --service-account=serml-api@serml-app.iam.gserviceaccount.com \
  --ingress=internal-and-cloud-load-balancing \
  --project=serml-app
```

### Manual Deploy (Frontend)

1. Connect `Skill-Based-Expert-Routing-System-For-Multilabel-Hate-Speech-Detection-main` to Cloudflare Pages
2. Build settings:
   - Build command: `cd web app/frontend && npm run build`
   - Output directory: `web app/frontend/dist`
3. Environment variables:
   - `VITE_API_BASE_URL` = Cloud Run service URL from above
4. Custom domain: `sberhsd.muditb0712.workers.dev`

### CI/CD (GitHub Actions)

The `.github/workflows/deploy.yml` handles automated deployment on push to `main`.

Required GitHub Secrets:
- `GCP_SA_KEY` — Service account JSON with Cloud Run Admin, Artifact Registry Writer, Secret Manager Accessor
- `CLOUDFLARE_API_TOKEN` — Pages deploy token
- `CLOUDFLARE_ACCOUNT_ID` — Your Cloudflare account ID

### Cost Estimate (Monthly)

| Component | Estimate |
|-----------|----------|
| Cloud Run GPU (L4, ~100 req/day, 30s avg) | $15-30/mo |
| Cloud Run CPU (scaled to 0) | $0 |
| Artifact Registry (10GB) | ~$0.50/mo |
| Cloudflare Pages | Free |
| **Total** | **~$15-30/mo (free for 10+ months with $300 credits)** |

### Local Development

```bash
# Backend
cd web app/backend
cp .env.example .env
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
cd web app/frontend
npm install
cp .env.example .env
# Set VITE_API_BASE_URL=http://localhost:8000 in .env
npm run dev
```
