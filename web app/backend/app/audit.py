from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


_STORAGE_DIR = Path(__file__).resolve().parents[1] / "storage"
_AUDIT_FILE = _STORAGE_DIR / "history.jsonl"


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_audit_event(*, text: str, response: dict[str, Any], latency_ms: float) -> None:
    _STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": response.get("timestamp"),
        "text_sha256": _text_hash(text),
        "text_length": len(text),
        "predicted_skills": response.get("predicted_skills", []),
        "output": response.get("output", []),
        "harmful": response.get("harmful", False),
        "risk_level": response.get("risk_level", "SAFE"),
        "latency_ms": latency_ms,
    }
    with _AUDIT_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
