# Colab GPU Backend Setup

Run these cells in Google Colab to host SERML FastAPI on GPU and connect local frontend.

## 1) Environment and dependencies

```bash
!nvidia-smi
!pip install -q -U pip
!pip install -q -r "web app/backend/requirements.txt" pyngrok
```

If your notebook does not already contain this repo, clone it first.

## 2) Configure backend environment

```python
import os
from pathlib import Path

# Adjust if your repo path differs
REPO_ROOT = Path.cwd()
BACKEND_DIR = REPO_ROOT / "web app" / "backend"

os.chdir(BACKEND_DIR)

os.environ["APP_HOST"] = "0.0.0.0"
os.environ["APP_PORT"] = "8000"
os.environ["MODEL_BACKEND"] = "local_inprocess"
os.environ["SYMBOLIC_MOE_DIR"] = str(REPO_ROOT / "symbolic-moe")
os.environ["ALLOW_ORIGINS"] = "http://localhost:5173,http://127.0.0.1:5173"
os.environ["MAX_INPUT_TOKENS"] = "1536"

# Optional if model is gated
# os.environ["HF_TOKEN"] = "hf_xxx"
```

## 3) Start ngrok tunnel and backend

```python
# Optional but recommended: set your ngrok authtoken
# os.environ["NGROK_AUTHTOKEN"] = "..."

from pyngrok import ngrok
import os
import uvicorn

authtoken = os.getenv("NGROK_AUTHTOKEN")
if authtoken:
    ngrok.set_auth_token(authtoken)

public_url = ngrok.connect(8000, "http").public_url
print("Public backend URL:", public_url)

uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
```

## 4) Point local frontend to Colab backend

In local machine:

```bash
cd "web app/frontend"
cp .env.example .env
```

Set `web app/frontend/.env`:

```bash
VITE_API_BASE_URL=https://YOUR-NGROK-URL
```

Then run frontend:

```bash
npm install
npm run dev
```

## Notes

- Colab sessions are temporary. URL changes each restart.
- Keep notebook runtime alive while using frontend.
- Backend startup loads the base Llama 3.1 model once, trains the NB top-2 router from `profile_pool_skills.jsonl`, and then reuses adapter switching for requests.
