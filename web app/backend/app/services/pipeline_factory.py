from __future__ import annotations

from app.config import settings
from app.services.base import InferenceService
from app.services.predict_post_adapter import PredictPostAdapterService
from app.services.remote_inference import RemoteInferenceService


def build_inference_service() -> InferenceService:
    if settings.model_backend == "local_inprocess":
        return PredictPostAdapterService()

    if settings.model_backend == "remote_inference":
        return RemoteInferenceService()

    raise ValueError(f"Unsupported MODEL_BACKEND '{settings.model_backend}'")
