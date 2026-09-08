# Colab GPU Backend Setup

Use this flow to run the FastAPI backend on Colab GPU for branch `skill-based-llm-driven-expert-routing-system-for-multilabel-harmful-speech-detection`.

The branch contains the backend and symbolic routing stack, while the deployed runtime artifacts continue to come from the GCS bucket:

- `saves/`
- `symbolic-moe/profiles.json`
- `symbolic-moe/skills.txt`

Notebook file in this branch:

- `web app/SERML_Backend.ipynb`

## 1. Clone the branch

```bash
!git clone -b skill-based-llm-driven-expert-routing-system-for-multilabel-harmful-speech-detection https://github.com/muditbaid/LLM-DoRA-Finetuning.git
%cd /content/LLM-DoRA-Finetuning
```

## 2. Install dependencies

```bash
!nvidia-smi
!pip install -q -U pip
!pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
!pip install -q -r "web app/backend/requirements.txt" pyngrok
```

## 3. Authenticate GCP and pull runtime artifacts

```python
from google.colab import auth

auth.authenticate_user()
```

```bash
!gcloud config set project serml-app
!gcloud storage cp -r gs://serml-app-artifacts/artifacts/saves /content/LLM-DoRA-Finetuning/
!gcloud storage cp gs://serml-app-artifacts/artifacts/symbolic-moe/profiles.json /content/LLM-DoRA-Finetuning/symbolic-moe/
!gcloud storage cp gs://serml-app-artifacts/artifacts/symbolic-moe/skills.txt /content/LLM-DoRA-Finetuning/symbolic-moe/
```

## 4. Configure backend environment and Hugging Face auth

```python
import getpass
import os
from pathlib import Path

from huggingface_hub import login

REPO_ROOT = Path("/content/LLM-DoRA-Finetuning")
BACKEND_DIR = REPO_ROOT / "web app" / "backend"

os.chdir(BACKEND_DIR)

os.environ["APP_HOST"] = "0.0.0.0"
os.environ["APP_PORT"] = "8000"
os.environ["MODEL_BACKEND"] = "local_inprocess"
os.environ["MODEL_PRELOAD"] = "true"
os.environ["MODEL_WARMUP"] = "true"
os.environ["SYMBOLIC_MOE_DIR"] = str(REPO_ROOT / "symbolic-moe")
os.environ["ALLOW_ORIGINS"] = "http://localhost:5173,http://127.0.0.1:5173,https://sberhsd.muditb0712.workers.dev"
os.environ["RUNS"] = "5"
os.environ["MIN_COUNT"] = "2"
os.environ["MAX_NEW_TOKENS"] = "64"
os.environ["MAX_INPUT_TOKENS"] = "1536"

hf_token = getpass.getpass("HF token: ")
login(hf_token)
os.environ["HF_TOKEN"] = hf_token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
```

## 5. Start backend, wait for readiness, then open ngrok

```python
import getpass
import subprocess
import time

import requests
from pyngrok import ngrok

REPO_ROOT = "/content/LLM-DoRA-Finetuning"
BACKEND_LOG = "/content/uvicorn.log"

subprocess.run("pkill -f 'uvicorn app.main:app' || true", shell=True, check=False)
ngrok.kill()

ngrok_auth_token = getpass.getpass("ngrok auth token: ")
ngrok.set_auth_token(ngrok_auth_token)

env = os.environ.copy()
logf = open(BACKEND_LOG, "w")
proc = subprocess.Popen(
    [
        "python",
        "-m",
        "uvicorn",
        "app.main:app",
        "--app-dir",
        "web app/backend",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--log-level",
        "info",
    ],
    cwd=REPO_ROOT,
    env=env,
    stdout=logf,
    stderr=subprocess.STDOUT,
)

deadline = time.time() + 900
while time.time() < deadline:
    if proc.poll() is not None:
        break
    try:
        response = requests.get("http://127.0.0.1:8000/health/ready", timeout=3)
        if response.ok:
            break
    except Exception:
        pass
    time.sleep(3)
else:
    raise RuntimeError("Backend did not become ready within 15 minutes.")

public_url = ngrok.connect(addr="127.0.0.1:8000", proto="http").public_url
print("Backend PID:", proc.pid)
print("Public URL:", public_url)
```

## 6. Verify health

```python
import json
import requests

headers = {"ngrok-skip-browser-warning": "1"}

local_ready = requests.get("http://127.0.0.1:8000/health/ready", timeout=10)
public_ready = requests.get(f"{public_url}/health/ready", headers=headers, timeout=20)

print("Local /health/ready")
print(json.dumps(local_ready.json(), indent=2))
print()
print("Public /health/ready")
print(json.dumps(public_ready.json(), indent=2))
```

## 7. Smoke-test detection

```python
import json
import requests

payload = {
    "text": "I will find you and hurt you."
}

response = requests.post(
    f"{public_url}/api/detect",
    json=payload,
    headers={"ngrok-skip-browser-warning": "1"},
    timeout=900,
)
response.raise_for_status()
print(json.dumps(response.json(), indent=2))
```

## Notes

- Colab sessions are temporary.
- With `MODEL_PRELOAD=true`, backend startup builds the NB top-2 router and
  loads the base Llama 3.1 model plus all expert adapters in a background
  thread. With `MODEL_WARMUP=true`, it also runs a real inference to exercise
  CUDA/Triton. `/health/ready` returns HTTP 503 until that work completes.
- The GCS bucket remains the source of truth for the deployed `profiles.json`.
