from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

from app.config import resolve_symbolic_moe_dir, settings
from app.schemas import DetectResponse, ExpertOutput, compute_risk_level, utc_now_iso
from app.services.base import InferenceService


LOGGER = logging.getLogger(__name__)


def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


class PredictPostAdapterService(InferenceService):
    """Local in-process wrapper around symbolic-moe/predict_post.py."""

    def __init__(self) -> None:
        self._ready = False
        self._lock = threading.Lock()

        self._symbolic_dir = resolve_symbolic_moe_dir()
        self._predict_mod: Any = None

        self._runtime: Any = None
        self._router: Any = None
        self._profiles: dict[str, Any] = {}
        self._experts_by_name: dict[str, Any] = {}
        self._args = argparse.Namespace(
            runs=settings.runs,
            min_count=settings.min_count,
            max_new_tokens=settings.max_new_tokens,
            max_input_tokens=settings.max_input_tokens,
        )

    def startup(self) -> None:
        if self._ready:
            return

        if settings.hf_token:
            os.environ.setdefault("HF_TOKEN", settings.hf_token)
            os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", settings.hf_token)

        predict_path = self._symbolic_dir / "predict_post.py"
        if not predict_path.exists():
            raise FileNotFoundError(f"predict_post.py not found at {predict_path}")

        self._predict_mod = _load_module(predict_path, "serml_predict_post_runtime")
        self._profiles = self._predict_mod.load_profiles()
        self._router = self._predict_mod.load_nb_router()

        self._runtime = self._predict_mod.SharedModelRuntime()
        self._experts_by_name = {}
        for cfg in self._predict_mod.EXPERTS:
            if cfg.name in self._profiles:
                self._experts_by_name[cfg.name] = self._predict_mod.ExpertModel(cfg, runtime=self._runtime)

        self._ready = True
        LOGGER.info(
            "Local symbolic-moe NB top-2 pipeline initialized with %d experts from %s",
            len(self._experts_by_name),
            self._symbolic_dir,
        )

    def shutdown(self) -> None:
        if not self._ready:
            return

        for model in self._experts_by_name.values():
            model.close()

        if self._runtime is not None:
            self._runtime.close()

        self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    def detect(self, text: str) -> DetectResponse:
        if not self._ready:
            raise RuntimeError("Inference service is not ready")

        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Input text must not be empty.")

        with self._lock:
            raw = self._predict_mod.run_single(
                cleaned,
                self._profiles,
                self._router,
                self._runtime,
                self._experts_by_name,
                self._args,
            )

        parsed_output = [ExpertOutput.model_validate(item) for item in raw.get("output", [])]
        risk_level = compute_risk_level(parsed_output)

        return DetectResponse(
            post=raw.get("post", cleaned),
            predicted_skills=list(raw.get("predicted_skills", [])),
            output=parsed_output,
            harmful=len(parsed_output) > 0,
            risk_level=risk_level,
            timestamp=utc_now_iso(),
        )
