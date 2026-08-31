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
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Notes

- The backend uses the local in-process NB top-2 pipeline by default.
- At startup it loads `profiles.json`, builds the Bernoulli router from `profile_pool_skills.jsonl`, and warms the shared Llama runtime once.
- The web API serves `POST /api/detect`, `GET /health/live`, and `GET /health/ready`.
