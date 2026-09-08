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
    """Local in-process wrapper around symbolic-moe/predict_post.py with lazy loading."""

    def __init__(self) -> None:
        self._ready = False
        self._models_loaded = False
        self._models_loading = False
        self._model_load_error: str | None = None
        self._model_thread: threading.Thread | None = None
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._shutdown_event = threading.Event()

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
        """Mark service as ready for health checks immediately. Models load lazily."""
        if self._ready:
            return

        self._shutdown_event.clear()

        if settings.hf_token:
            os.environ.setdefault("HF_TOKEN", settings.hf_token)
            os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", settings.hf_token)

        predict_path = self._symbolic_dir / "predict_post.py"
        if not predict_path.exists():
            raise FileNotFoundError(f"predict_post.py not found at {predict_path}")

        self._predict_mod = _load_module(predict_path, "serml_predict_post_runtime")
        self._profiles = self._predict_mod.load_profiles()
        missing_profiles = [
            cfg.name for cfg in self._predict_mod.EXPERTS if cfg.name not in self._profiles
        ]
        if missing_profiles:
            raise RuntimeError(
                "Missing expert profiles: " + ", ".join(sorted(missing_profiles))
            )
        self._router = self._predict_mod.load_nb_router()

        # The process is live; model readiness remains false until loading and
        # the optional production warmup complete.
        self._ready = True
        LOGGER.info(
            "Service initialized; model_preload=%s model_warmup=%s",
            settings.model_preload,
            settings.model_warmup,
        )

        if settings.model_preload:
            self._model_thread = threading.Thread(
                target=self._preload_models,
                name="serml-model-preload",
                daemon=True,
            )
            self._model_thread.start()

    def _preload_models(self) -> None:
        try:
            self._ensure_models_loaded()
        except Exception:  # noqa: BLE001
            # _ensure_models_loaded records and logs the full failure. A later
            # detection request may retry without taking down the web server.
            return

    def _warmup_models(self, runtime: Any, experts: dict[str, Any]) -> None:
        """Exercise base inference, routing, and every configured adapter."""
        warmup_post = "System warmup request."
        warmup_args = argparse.Namespace(
            runs=1,
            min_count=1,
            max_new_tokens=min(settings.max_new_tokens, 16),
            max_input_tokens=min(settings.max_input_tokens, 128),
        )
        result = self._predict_mod.run_single(
            warmup_post,
            self._profiles,
            self._router,
            runtime,
            experts,
            warmup_args,
        )
        warmed_experts = set(result.get("routed_experts", []))

        for cfg in self._predict_mod.EXPERTS:
            if cfg.name in warmed_experts:
                continue
            if self._shutdown_event.is_set():
                raise RuntimeError("Model warmup canceled during shutdown")

            expert = experts[cfg.name]
            system, instruction = self._predict_mod.build_expert_prompt_parts(cfg.label)
            prompt = expert.build_prompt(system, instruction, warmup_post)
            expert.predict_with_confidence(prompt)
            warmed_experts.add(cfg.name)

        expected_experts = set(experts)
        if warmed_experts != expected_experts:
            missing = sorted(expected_experts - warmed_experts)
            raise RuntimeError("Warmup did not exercise experts: " + ", ".join(missing))

        LOGGER.info(
            "End-to-end CUDA/Triton warmup inference completed for experts=%s",
            ",".join(sorted(warmed_experts)),
        )

    def _ensure_models_loaded(self) -> None:
        """Load models once, waiting for an in-progress background preload."""
        if self._models_loaded:
            return

        with self._model_lock:
            if self._models_loaded:
                return
            if self._shutdown_event.is_set():
                raise RuntimeError("Inference service is shutting down")

            self._models_loading = True
            self._model_load_error = None
            runtime: Any = None
            experts: dict[str, Any] = {}
            try:
                LOGGER.info("Loading base model and expert adapters...")
                runtime = self._predict_mod.SharedModelRuntime()
                for cfg in self._predict_mod.EXPERTS:
                    if cfg.name in self._profiles:
                        experts[cfg.name] = self._predict_mod.ExpertModel(cfg, runtime=runtime)

                if settings.model_warmup:
                    LOGGER.info("Running end-to-end CUDA/Triton warmup inference...")
                    self._warmup_models(runtime, experts)

                if self._shutdown_event.is_set():
                    raise RuntimeError("Model initialization canceled during shutdown")

                self._runtime = runtime
                self._experts_by_name = experts
                self._models_loaded = True
                LOGGER.info(
                    "Local symbolic-moe NB top-2 pipeline initialized with %d experts from %s",
                    len(self._experts_by_name),
                    self._symbolic_dir,
                )
            except Exception as exc:  # noqa: BLE001
                if not self._shutdown_event.is_set():
                    self._model_load_error = f"{type(exc).__name__}: {exc}"
                LOGGER.exception("Model initialization failed")
                if runtime is not None:
                    try:
                        runtime.close()
                    except Exception:  # noqa: BLE001
                        LOGGER.exception("Failed to release partially loaded model runtime")
                raise RuntimeError("Model initialization failed") from exc
            finally:
                self._models_loading = False

    def shutdown(self) -> None:
        self._shutdown_event.set()
        # Synchronize with preload so it cannot publish a runtime after the
        # service state has already been cleared.
        with self._model_lock:
            with self._inference_lock:
                for model in self._experts_by_name.values():
                    try:
                        model.close()
                    except Exception:  # noqa: BLE001
                        LOGGER.exception("Failed to close expert")

                if self._runtime is not None:
                    self._runtime.close()

                self._runtime = None
                self._experts_by_name = {}
                self._ready = False
                self._models_loaded = False
                self._models_loading = False
                self._model_thread = None

    def is_ready(self) -> bool:
        return self._ready

    def is_model_ready(self) -> bool:
        return self._models_loaded

    def model_status(self) -> str:
        if self._models_loaded:
            return "ready"
        if self._models_loading:
            return "loading"
        if self._model_load_error:
            return "error"
        return "not_loaded"

    def detect(self, text: str) -> DetectResponse:
        if not self._ready:
            raise RuntimeError("Inference service is not ready")

        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Input text must not be empty.")

        # If background preload is still running, the first request waits here.
        # If preload failed, this call retries and returns a clear server error.
        self._ensure_models_loaded()

        with self._inference_lock:
            # Shutdown takes the model lock and then this inference lock. It can
            # therefore clear a runtime after the load check above while this
            # request is waiting here; revalidate without reversing lock order.
            if (
                self._shutdown_event.is_set()
                or not self._ready
                or not self._models_loaded
                or self._runtime is None
            ):
                raise RuntimeError("Inference service shut down before execution")
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
