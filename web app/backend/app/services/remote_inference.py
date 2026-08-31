from __future__ import annotations

import logging

import httpx

from app.config import settings
from app.schemas import DetectResponse
from app.services.base import InferenceService


LOGGER = logging.getLogger(__name__)


class RemoteInferenceService(InferenceService):
    def __init__(self) -> None:
        if not settings.gpu_endpoint_url:
            raise ValueError("GPU_ENDPOINT_URL is required for remote_inference mode.")

        self._url = settings.gpu_endpoint_url.rstrip("/") + "/api/detect"
        self._timeout = settings.gpu_endpoint_timeout_seconds
        self._ready = False

    def startup(self) -> None:
        self._ready = True
        LOGGER.info("Remote inference service configured at %s", self._url)

    def shutdown(self) -> None:
        self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    def detect(self, text: str) -> DetectResponse:
        payload = {"text": text}
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(self._url, json=payload)
            response.raise_for_status()
            data = response.json()
        return DetectResponse.model_validate(data)
