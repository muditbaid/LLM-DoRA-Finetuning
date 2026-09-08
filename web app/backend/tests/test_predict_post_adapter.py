from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services import predict_post_adapter as adapter  # noqa: E402


class _FakeRuntime:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _FakeExpert:
    def __init__(self, name: str, runtime: _FakeRuntime) -> None:
        self.name = name
        self.runtime = runtime
        self.close_calls = 0
        self.predict_calls = 0

    def close(self) -> None:
        self.close_calls += 1

    def build_prompt(self, system: str, instruction: str, post: str) -> str:
        return f"{system}\n{instruction}\n{post}"

    def predict_with_confidence(self, prompt: str):
        del prompt
        self.predict_calls += 1
        return "safe", 1.0


class _FakePredictModule:
    """Small stand-in for predict_post.py that never imports the ML stack."""

    def __init__(
        self,
        *,
        expert_names: tuple[str, ...] = ("expert-a", "expert-b"),
        on_runtime_created=None,
        on_expert_created=None,
        on_run_single=None,
    ) -> None:
        self.EXPERTS = tuple(
            SimpleNamespace(name=name, label=f"{name}-label")
            for name in expert_names
        )
        self._on_runtime_created = on_runtime_created
        self._on_expert_created = on_expert_created
        self._on_run_single = on_run_single
        self._state_lock = threading.Lock()

        self.runtime_calls = 0
        self.expert_calls = 0
        self.run_single_calls = 0
        self.run_single_texts: list[str] = []
        self.runtimes: list[_FakeRuntime] = []
        self.experts: list[_FakeExpert] = []

        # The production adapter treats these as module attributes and calls them.
        self.SharedModelRuntime = self._make_runtime
        self.ExpertModel = self._make_expert

    def load_profiles(self) -> dict[str, object]:
        return {config.name: object() for config in self.EXPERTS}

    def load_nb_router(self) -> object:
        return object()

    def build_expert_prompt_parts(self, label: str) -> tuple[str, str]:
        return f"system for {label}", f"instruction for {label}"

    def _make_runtime(self) -> _FakeRuntime:
        runtime = _FakeRuntime()
        with self._state_lock:
            self.runtime_calls += 1
            self.runtimes.append(runtime)
        if self._on_runtime_created is not None:
            self._on_runtime_created(runtime)
        return runtime

    def _make_expert(
        self, config: SimpleNamespace, *, runtime: _FakeRuntime
    ) -> _FakeExpert:
        expert = _FakeExpert(config.name, runtime)
        with self._state_lock:
            self.expert_calls += 1
            call_number = self.expert_calls
            self.experts.append(expert)
        if self._on_expert_created is not None:
            self._on_expert_created(expert, call_number)
        return expert

    def run_single(
        self,
        text: str,
        profiles,
        router,
        runtime,
        experts,
        args,
    ) -> dict[str, object]:
        del profiles, router, runtime, experts, args
        with self._state_lock:
            self.run_single_calls += 1
            self.run_single_texts.append(text)
        if self._on_run_single is not None:
            self._on_run_single(text)
        return {"post": text, "predicted_skills": [], "output": []}


class PredictPostAdapterLifecycleTests(unittest.TestCase):
    def _make_service(
        self,
        predict_module: _FakePredictModule,
        *,
        model_preload: bool,
        model_warmup: bool,
    ) -> adapter.PredictPostAdapterService:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        symbolic_dir = Path(temporary_directory.name)
        (symbolic_dir / "predict_post.py").write_text(
            "# Replaced by the fake module in lifecycle tests.\n", encoding="utf-8"
        )

        test_settings = replace(
            adapter.settings,
            symbolic_moe_dir=str(symbolic_dir),
            model_preload=model_preload,
            model_warmup=model_warmup,
        )
        settings_patch = patch.object(adapter, "settings", test_settings)
        directory_patch = patch.object(
            adapter, "resolve_symbolic_moe_dir", return_value=symbolic_dir
        )
        loader_patch = patch.object(
            adapter, "_load_module", return_value=predict_module
        )
        settings_patch.start()
        directory_patch.start()
        loader_patch.start()
        self.addCleanup(settings_patch.stop)
        self.addCleanup(directory_patch.stop)
        self.addCleanup(loader_patch.stop)

        service = adapter.PredictPostAdapterService()
        self.addCleanup(service.shutdown)
        return service

    def test_model_not_ready_until_warmup_succeeds(self) -> None:
        warmup_started = threading.Event()
        allow_warmup_to_finish = threading.Event()

        def block_warmup(text: str) -> None:
            if text == "System warmup request.":
                warmup_started.set()
                if not allow_warmup_to_finish.wait(timeout=5):
                    raise TimeoutError("test did not release warmup")

        predict_module = _FakePredictModule(on_run_single=block_warmup)
        service = self._make_service(
            predict_module, model_preload=True, model_warmup=True
        )

        service.startup()
        preload_thread = service._model_thread
        self.assertIsNotNone(preload_thread)
        try:
            self.assertTrue(warmup_started.wait(timeout=2))
            self.assertFalse(service.is_model_ready())
            self.assertEqual("loading", service.model_status())
        finally:
            allow_warmup_to_finish.set()
            if preload_thread is not None:
                preload_thread.join(timeout=5)

        self.assertIsNotNone(preload_thread)
        self.assertFalse(preload_thread.is_alive())
        self.assertTrue(service.is_model_ready())
        self.assertEqual("ready", service.model_status())
        self.assertEqual(["System warmup request."], predict_module.run_single_texts)
        self.assertTrue(
            all(expert.predict_calls == 1 for expert in predict_module.experts)
        )

    def test_initialization_failure_is_wrapped_and_sets_error_status(self) -> None:
        def fail_runtime_initialization(runtime: _FakeRuntime) -> None:
            del runtime
            raise ValueError("CUDA runtime unavailable")

        predict_module = _FakePredictModule(
            on_runtime_created=fail_runtime_initialization
        )
        service = self._make_service(
            predict_module, model_preload=False, model_warmup=True
        )
        service.startup()

        with self.assertRaisesRegex(
            RuntimeError, "^Model initialization failed$"
        ) as raised:
            service._ensure_models_loaded()

        self.assertIsInstance(raised.exception.__cause__, ValueError)
        self.assertEqual("CUDA runtime unavailable", str(raised.exception.__cause__))
        self.assertFalse(service.is_model_ready())
        self.assertEqual("error", service.model_status())
        self.assertEqual(
            "ValueError: CUDA runtime unavailable", service._model_load_error
        )

    def test_concurrent_detect_calls_initialize_models_once(self) -> None:
        initialization_started = threading.Event()
        allow_initialization_to_finish = threading.Event()

        def block_runtime_initialization(runtime: _FakeRuntime) -> None:
            del runtime
            initialization_started.set()
            if not allow_initialization_to_finish.wait(timeout=5):
                raise TimeoutError("test did not release initialization")

        predict_module = _FakePredictModule(
            on_runtime_created=block_runtime_initialization
        )
        service = self._make_service(
            predict_module, model_preload=False, model_warmup=False
        )
        service.startup()

        first_call_entered = threading.Event()
        second_call_entered = threading.Event()

        def detect(text: str, entered: threading.Event):
            entered.set()
            return service.detect(text)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(detect, "first request", first_call_entered)
            second = None
            try:
                self.assertTrue(first_call_entered.wait(timeout=2))
                self.assertTrue(initialization_started.wait(timeout=2))
                second = executor.submit(detect, "second request", second_call_entered)
                self.assertTrue(second_call_entered.wait(timeout=2))
                self.assertEqual(1, predict_module.runtime_calls)
            finally:
                allow_initialization_to_finish.set()

            first_result = first.result(timeout=5)
            self.assertIsNotNone(second)
            second_result = second.result(timeout=5)

        self.assertEqual("first request", first_result.post)
        self.assertEqual("second request", second_result.post)
        self.assertEqual(1, predict_module.runtime_calls)
        self.assertEqual(len(predict_module.EXPERTS), predict_module.expert_calls)
        self.assertEqual(2, predict_module.run_single_calls)
        self.assertTrue(service.is_model_ready())

    def test_shutdown_during_preload_cannot_publish_loaded_state(self) -> None:
        expert_load_started = threading.Event()
        allow_expert_load_to_finish = threading.Event()

        def block_first_expert(expert: _FakeExpert, call_number: int) -> None:
            del expert
            if call_number == 1:
                expert_load_started.set()
                if not allow_expert_load_to_finish.wait(timeout=5):
                    raise TimeoutError("test did not release expert loading")

        predict_module = _FakePredictModule(
            expert_names=("expert-a",), on_expert_created=block_first_expert
        )
        service = self._make_service(
            predict_module, model_preload=True, model_warmup=False
        )
        service.startup()
        preload_thread = service._model_thread
        self.assertIsNotNone(preload_thread)

        shutdown_finished = threading.Event()
        shutdown_errors: list[BaseException] = []

        def shut_down() -> None:
            try:
                service.shutdown()
            except BaseException as exc:  # pragma: no cover - asserted below
                shutdown_errors.append(exc)
            finally:
                shutdown_finished.set()

        shutdown_thread = threading.Thread(target=shut_down, name="test-shutdown")
        try:
            self.assertTrue(expert_load_started.wait(timeout=2))
            shutdown_thread.start()
            self.assertTrue(service._shutdown_event.wait(timeout=2))
            self.assertFalse(shutdown_finished.is_set())
        finally:
            allow_expert_load_to_finish.set()
            if preload_thread is not None:
                preload_thread.join(timeout=5)
            if shutdown_thread.ident is not None:
                shutdown_thread.join(timeout=5)

        self.assertFalse(shutdown_thread.is_alive())
        self.assertIsNotNone(preload_thread)
        self.assertFalse(preload_thread.is_alive())
        self.assertEqual([], shutdown_errors)
        self.assertFalse(service.is_ready())
        self.assertFalse(service.is_model_ready())
        self.assertEqual("not_loaded", service.model_status())
        self.assertIsNone(service._runtime)
        self.assertEqual({}, service._experts_by_name)
        self.assertEqual(1, predict_module.runtime_calls)
        self.assertEqual(1, predict_module.runtimes[0].close_calls)

    def test_detect_revalidates_runtime_after_shutdown_race(self) -> None:
        predict_module = _FakePredictModule(expert_names=("expert-a",))
        service = self._make_service(
            predict_module, model_preload=False, model_warmup=False
        )
        service.startup()
        service._ensure_models_loaded()

        load_check_finished = threading.Event()
        allow_detect_to_continue = threading.Event()
        original_ensure_models_loaded = service._ensure_models_loaded

        def pause_after_load_check() -> None:
            original_ensure_models_loaded()
            load_check_finished.set()
            if not allow_detect_to_continue.wait(timeout=5):
                raise TimeoutError("test did not release detection")

        detect_errors: list[BaseException] = []

        def detect() -> None:
            try:
                service.detect("request racing with shutdown")
            except BaseException as exc:  # pragma: no branch - asserted below
                detect_errors.append(exc)

        with patch.object(service, "_ensure_models_loaded", pause_after_load_check):
            detect_thread = threading.Thread(target=detect, name="test-detect")
            detect_thread.start()
            try:
                self.assertTrue(load_check_finished.wait(timeout=2))
                service.shutdown()
            finally:
                allow_detect_to_continue.set()
                detect_thread.join(timeout=5)

        self.assertFalse(detect_thread.is_alive())
        self.assertEqual(1, len(detect_errors))
        self.assertIsInstance(detect_errors[0], RuntimeError)
        self.assertEqual(0, predict_module.run_single_calls)


if __name__ == "__main__":
    unittest.main()
