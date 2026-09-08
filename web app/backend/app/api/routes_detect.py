from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.audit import write_audit_event
from app.config import settings
from app.schemas import DetectRequest, DetectResponse, HealthResponse, utc_now_iso


LOGGER = logging.getLogger(__name__)
router = APIRouter()

limiter = Limiter(key_func=get_remote_address)


def _service(request: Request):
    return request.app.state.inference_service


@router.get("/health/live", response_model=HealthResponse)
async def health_live(request: Request) -> HealthResponse:
    service = _service(request)
    return HealthResponse(
        status="ok",
        backend=settings.model_backend,
        model_ready=service.is_model_ready(),
        timestamp=utc_now_iso(),
    )


@router.get("/health/ready", response_model=HealthResponse)
async def health_ready(request: Request, response: Response) -> HealthResponse:
    service = _service(request)
    model_ready = service.is_model_ready()
    if not model_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if model_ready else service.model_status(),
        backend=settings.model_backend,
        model_ready=model_ready,
        timestamp=utc_now_iso(),
    )


@router.post("/api/detect", response_model=DetectResponse)
@limiter.limit("10/minute")
async def detect(payload: DetectRequest, request: Request) -> DetectResponse:
    service = _service(request)
    if not service.is_ready():
        raise HTTPException(status_code=503, detail="Inference backend is not ready")

    start = time.perf_counter()
    LOGGER.info("Detection request received: text_len=%d", len(payload.text))
    try:
        result = await run_in_threadpool(service.detect, payload.text)
        LOGGER.info("Detection succeeded: text_len=%d", len(payload.text))
    except ValueError as exc:
        LOGGER.info("Invalid detection request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        LOGGER.exception("Inference backend unavailable")
        raise HTTPException(status_code=503, detail="Inference backend unavailable") from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Detection failed")
        raise HTTPException(status_code=500, detail="Detection failed") from exc

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    LOGGER.info(
        "detect_completed text_len=%d skills=%d labels=%d harmful=%s latency_ms=%.2f",
        len(payload.text),
        len(result.predicted_skills),
        len(result.output),
        result.harmful,
        duration_ms,
    )
    await run_in_threadpool(
        write_audit_event,
        text=payload.text,
        response=result.model_dump(),
        latency_ms=duration_ms,
    )
    return result
