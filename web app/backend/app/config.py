from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "SERML API")
    app_env: str = os.getenv("APP_ENV", "dev")
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    app_log_level: str = os.getenv("APP_LOG_LEVEL", "INFO")

    allow_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv(
            "ALLOW_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    )

    model_backend: str = os.getenv("MODEL_BACKEND", "local_inprocess")
    symbolic_moe_dir: str = os.getenv("SYMBOLIC_MOE_DIR", "../../symbolic-moe")

    hf_token: str | None = os.getenv("HF_TOKEN") or None

    runs: int = int(os.getenv("RUNS", "5"))
    min_count: int = int(os.getenv("MIN_COUNT", "2"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "64"))

    gpu_endpoint_url: str | None = os.getenv("GPU_ENDPOINT_URL") or None
    gpu_endpoint_timeout_seconds: int = int(os.getenv("GPU_ENDPOINT_TIMEOUT_SECONDS", "45"))


settings = Settings()


def resolve_symbolic_moe_dir() -> Path:
    configured = Path(settings.symbolic_moe_dir)
    if configured.is_absolute():
        return configured

    backend_dir = Path(__file__).resolve().parents[2]
    return (backend_dir / configured).resolve()
