# SERML Web App

FastAPI backend + single-file React frontend for Symbolic-MoE inference.

## Structure

- `web app/backend/`: API server, model runtime integration, audit logging
- `web app/frontend/SERMLApp.jsx`: single-file React UI (Tailwind + Recharts)

## Backend

### Requirements

Install backend dependencies:

```bash
cd "web app/backend"
pip install -r requirements.txt
```

### Environment

```bash
cp .env.example .env
```

Key variables:

- `MODEL_BACKEND=local_inprocess|remote_inference`
- `SYMBOLIC_MOE_DIR` path to `symbolic-moe`
- `RUNS`, `MIN_COUNT`, `MAX_NEW_TOKENS` match `predict_post.py`
- `GPU_ENDPOINT_URL` for remote backend mode

### Run

```bash
cd "web app/backend"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Endpoints

- `POST /api/detect`
  - Request: `{ "text": "..." }`
  - Response: same core shape as `predict_post.py` with extra fields:
    - `harmful`
    - `risk_level`
    - `timestamp`
- `GET /health/live`
- `GET /health/ready`

## Frontend (Local)

```bash
cd "web app/frontend"
cp .env.example .env
npm install
npm run dev
```

- Local backend mode:
  - Keep `VITE_API_BASE_URL` empty in `web app/frontend/.env`
  - Vite proxy sends `/api/*` to `http://127.0.0.1:8000`
- Colab backend mode:
  - Set `VITE_API_BASE_URL` to your public Colab tunnel URL (no trailing slash)
  - Example: `VITE_API_BASE_URL=https://abcd-12-34-56-78.ngrok-free.app`

## Data Dependencies

`MODEL_BACKEND=local_inprocess` expects existing artifacts in project root:

- `symbolic-moe/profiles.json`
- `symbolic-moe/skills.txt`
- adapters from `symbolic-moe/config.py` under `saves/...`
- base model access (`meta-llama/Meta-Llama-3.1-8B-Instruct`)

## Storage and Logs

- API logs: stdout JSON logs (ship to cloud logging)
- Local audit log: `web app/backend/storage/history.jsonl`
  - Stores text hash and length, not full input text
- Runtime log folder placeholder: `web app/backend/logs/`

## Frontend

`web app/frontend/SERMLApp.jsx` is framework-agnostic React component. Mount it in your existing React app and ensure Tailwind + Recharts are available.

Expected backend path: `/api/detect`.

## Colab Backend (GPU)

See `web app/COLAB_BACKEND.md` for notebook cell commands to:

- install backend deps
- run FastAPI with `MODEL_BACKEND=local_inprocess`
- expose API URL publicly via ngrok
- connect local frontend to that URL
