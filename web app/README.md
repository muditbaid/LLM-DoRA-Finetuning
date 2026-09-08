# Web App

This directory contains the deployment-side assets for the system web experience.

## Included

- `backend/`: FastAPI service that wraps `symbolic-moe/predict_post.py`
- `COLAB_BACKEND.md`: Colab GPU setup instructions for temporary backend hosting
- `SERML_Backend.ipynb`: runnable Colab notebook for backend startup and tunnel exposure

## Hosted UI

[Web UI](https://sberhsd.muditb0712.workers.dev)

## Backend Start

```bash
cd "web app/backend"
cp .env.example .env
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

See [the production deployment guide](../docs/PRODUCTION_DEPLOYMENT.md) for
Cloud Build, Cloud Run GPU, direct IAP, and Cloudflare Workers instructions.

## Notes

- The backend uses the local in-process NB top-2 pipeline by default.
- At startup it loads `profiles.json` and builds the Bernoulli router from
  `profile_pool_skills.jsonl`. The shared Llama runtime loads on demand locally
  or in a background thread when `MODEL_PRELOAD=true` in production. With
  `MODEL_WARMUP=true`, readiness waits for real inference through every adapter.
- The web API serves `POST /api/detect`, `GET /health/live`, and `GET /health/ready`.
