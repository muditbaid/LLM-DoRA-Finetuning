from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_detect import router as detect_router
from app.config import settings
from app.logging_config import configure_logging
from app.services.pipeline_factory import build_inference_service


configure_logging()
LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = build_inference_service()
    app.state.inference_service = service
    service.startup()
    LOGGER.info("API startup complete backend=%s", settings.model_backend)
    try:
        yield
    finally:
        service.shutdown()
        LOGGER.info("API shutdown complete")


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allow_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(detect_router)
