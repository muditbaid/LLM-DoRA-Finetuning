from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.api.routes_detect import router as detect_router
from app.config import settings
from app.logging_config import configure_logging
from app.services.pipeline_factory import build_inference_service


configure_logging()
LOGGER = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = build_inference_service()
    app.state.inference_service = service
    app.state.limiter = limiter
    service.startup()
    LOGGER.info("API startup complete backend=%s", settings.model_backend)
    try:
        yield
    finally:
        service.shutdown()
        LOGGER.info("API shutdown complete")


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allow_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(detect_router)


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint (placeholder for future instrumentation)."""
    return JSONResponse(content={"status": "ok", "message": "Metrics endpoint - integrate prometheus-fastapi-instrumentator for full metrics"})
