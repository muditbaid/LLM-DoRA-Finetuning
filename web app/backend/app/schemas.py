from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DetectRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


class ExpertOutput(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class DetectResponse(BaseModel):
    post: str
    predicted_skills: list[str]
    output: list[ExpertOutput]
    harmful: bool
    risk_level: Literal["SAFE", "LOW", "MEDIUM", "HIGH"]
    timestamp: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    backend: str
    model_ready: bool
    timestamp: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_risk_level(output: list[ExpertOutput]) -> str:
    if not output:
        return "SAFE"

    top = max(item.confidence for item in output)
    if top >= 0.85:
        return "HIGH"
    if top >= 0.65:
        return "MEDIUM"
    return "LOW"
