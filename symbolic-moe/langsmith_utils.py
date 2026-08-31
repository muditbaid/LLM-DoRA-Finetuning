"""
Optional LangSmith helpers for symbolic-moe workflows.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from langsmith import Client
    from langsmith.run_trees import RunTree
except ImportError:  # pragma: no cover - optional dependency
    Client = None  # type: ignore[assignment]
    RunTree = None  # type: ignore[assignment]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _safe_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json(v) for v in value]
    return str(value)


def file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def git_commit(cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            check=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def infer_split(path: Path | None) -> str:
    if path is None:
        return "unknown"
    name = path.name.lower()
    if "validation" in name or "val" in name:
        return "validation"
    if "test" in name:
        return "test"
    if "profile" in name:
        return "profile"
    return "unknown"


def tracing_enabled() -> bool:
    return bool(Client is not None and _env_flag("LANGSMITH_TRACING", default=False))


def parse_tags(raw_tags: str | None) -> list[str]:
    if not raw_tags:
        return []
    seen: set[str] = set()
    tags: list[str] = []
    for part in raw_tags.split(","):
        tag = part.strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def task_label_space(label_texts: Sequence[str]) -> list[str]:
    return [text.strip() for text in label_texts if isinstance(text, str) and text.strip()]


def sample_key_for_row(row: dict[str, Any]) -> str:
    return f"{row.get('dataset')}::{row.get('id')}"


def root_trace_inputs(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset"),
        "input": row.get("input", ""),
    }


def format_keyword_responses(keyword_responses: Any) -> dict[str, Any]:
    if isinstance(keyword_responses, dict):
        return {str(k): _safe_json(v) for k, v in keyword_responses.items()}
    if not isinstance(keyword_responses, list):
        return {}
    return {
        f"run_{index}": _safe_json(response)
        for index, response in enumerate(keyword_responses)
    }


def skill_stage_outputs(
    *,
    predicted_skills: Any,
    skill_vote_counts: Any,
    keyword_responses: Any,
    unselected_skills: Any,
    empty_skills: bool,
) -> dict[str, Any]:
    outputs = {
        "predicted_skills": _safe_json(predicted_skills or []),
        "skill_vote_counts": _safe_json(skill_vote_counts or {}),
        "keyword_responses": format_keyword_responses(keyword_responses),
        "unselected_skills": _safe_json(unselected_skills or []),
        "empty_skills": empty_skills,
    }
    return outputs


def default_dataset_name_for_split(split: str) -> str | None:
    if split == "validation":
        return "symbolic-moe-validation"
    if split == "test":
        return "symbolic-moe-test"
    return None


class LangSmithManager:
    def __init__(
        self,
        project_name: str | None = None,
        tags: Iterable[str] | None = None,
        allow_without_tracing: bool = False,
    ):
        self.enabled = tracing_enabled()
        self.can_trace = self.enabled or allow_without_tracing
        self.project_name = project_name or os.getenv("LANGSMITH_PROJECT", "symbolic-moe")
        self.tags = list(tags or [])
        can_create_client = Client is not None and self.can_trace
        self.client = Client() if can_create_client else None
        self._dataset_example_cache: dict[str, dict[str, str]] = {}

    def create_root(
        self,
        *,
        name: str,
        inputs: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        tags: Iterable[str] | None = None,
        reference_example_id: str | None = None,
    ):
        if not self.can_trace or RunTree is None:
            return None

        root = RunTree(
            name=name,
            run_type="chain",
            inputs=_safe_json(inputs),
            reference_example_id=reference_example_id,
            extra={"metadata": _safe_json(metadata or {})},
            tags=list(dict.fromkeys([*self.tags, *(tags or [])])),
            project_name=self.project_name,
            ls_client=self.client,
        )
        root.post()
        if outputs is not None:
            root.end(outputs=_safe_json(outputs))
            root.patch()
        return root

    def create_child(
        self,
        parent,
        *,
        name: str,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        run_type: str = "tool",
        tags: Iterable[str] | None = None,
    ):
        if parent is None:
            return None
        child = parent.create_child(
            name=name,
            run_type=run_type,
            inputs=_safe_json(inputs or {}),
            extra={"metadata": _safe_json(metadata or {})},
            tags=list(tags or []),
        )
        child.post()
        if outputs is not None:
            child.end(outputs=_safe_json(outputs))
            child.patch()
        return child

    def end_run(self, run, *, outputs: dict[str, Any] | None = None, error: str | None = None) -> None:
        if run is None:
            return
        if error is not None:
            run.end(error=error, outputs=_safe_json(outputs or {}))
        else:
            run.end(outputs=_safe_json(outputs or {}))
        run.patch()

    def record_skill_example(
        self,
        *,
        row: dict[str, Any],
        annotation: dict[str, Any],
        metadata: dict[str, Any],
        tags: Iterable[str] | None = None,
        reference_example_id: str | None = None,
    ) -> str | None:
        sample_key = sample_key_for_row(row)
        root = self.create_root(
            name="symbolic_moe_example",
            inputs=root_trace_inputs(row),
            metadata={
                **metadata,
                "sample_id": row.get("id"),
                "sample_key": sample_key,
                "comparison_group": sample_key,
            },
            tags=tags,
            reference_example_id=reference_example_id,
        )
        self.create_child(
            root,
            name="skill_inference",
            inputs={"input": row.get("input", "")},
            outputs=skill_stage_outputs(
                predicted_skills=annotation.get("predicted_skills", []),
                skill_vote_counts=annotation.get("skill_vote_counts", {}),
                keyword_responses=annotation.get("keyword_responses", []),
                unselected_skills=annotation.get("unselected_skills", []),
                empty_skills=not bool(annotation.get("predicted_skills")),
            ),
            metadata={"stage": "skill_inference"},
        )
        self.end_run(
            root,
            outputs={
                "predicted_skills": annotation.get("predicted_skills", []),
                "skill_vote_counts": annotation.get("skill_vote_counts", {}),
                "empty_skills": not bool(annotation.get("predicted_skills")),
            },
        )
        return str(root.id) if root is not None else None

    def record_pipeline_example(
        self,
        *,
        row: dict[str, Any],
        metadata: dict[str, Any],
        skill_info: dict[str, Any],
        router_info: dict[str, Any],
        predictions: list[dict[str, Any]],
        evaluation: dict[str, Any],
        tags: Iterable[str] | None = None,
        reference_example_id: str | None = None,
    ) -> str | None:
        sample_key = sample_key_for_row(row)
        root = self.create_root(
            name="symbolic_moe_example",
            inputs=root_trace_inputs(row),
            metadata={
                **metadata,
                "sample_id": row.get("id"),
                "sample_key": sample_key,
                "comparison_group": sample_key,
            },
            tags=tags,
            reference_example_id=reference_example_id,
        )
        self.create_child(
            root,
            name="skill_inference",
            inputs={"input": row.get("input", "")},
            outputs=skill_stage_outputs(
                predicted_skills=skill_info.get("predicted_skills", []),
                skill_vote_counts=skill_info.get("skill_vote_counts", {}),
                keyword_responses=skill_info.get("keyword_responses", []),
                unselected_skills=skill_info.get("unselected_skills", []),
                empty_skills=skill_info.get("empty_skills", False),
            ),
            metadata={"stage": "skill_inference"},
        )
        self.create_child(
            root,
            name="router_scoring",
            inputs={"predicted_skills": skill_info.get("predicted_skills", [])},
            outputs={
                "score_map": router_info.get("score_map", {}),
                "ranked_experts": router_info.get("ranked_experts", []),
                "selected_experts": router_info.get("selected_experts", []),
                "routing_margin": router_info.get("routing_margin"),
                "selection_policy": router_info.get("selection_policy"),
            },
            metadata={"stage": "router"},
        )
        for prediction in predictions:
            self.create_child(
                root,
                name="expert_inference",
                inputs={
                    "expert": prediction.get("expert"),
                    "prompt": prediction.get("prompt"),
                    "task_label_space": prediction.get("task_label_space", []),
                    "routed_weight": prediction.get("weight"),
                },
                outputs={
                    "raw_prediction": prediction.get("raw_prediction"),
                    "normalized_prediction": prediction.get("normalized_prediction"),
                    "label_confidence": prediction.get("label_confidence"),
                    "latency_seconds": prediction.get("latency_seconds"),
                },
                metadata={
                    "stage": "expert_inference",
                    "expert": prediction.get("expert"),
                },
            )
        self.create_child(
            root,
            name="evaluation",
            inputs={
                "gold_label": row.get("output", ""),
                "dataset": row.get("dataset", ""),
            },
            outputs=evaluation,
            metadata={"stage": "evaluation"},
        )
        self.end_run(
            root,
            outputs={
                "predictions": predictions,
                "evaluation": evaluation,
                "selected_experts": router_info.get("selected_experts", []),
            },
        )
        return str(root.id) if root is not None else None

    def record_evaluation_example(
        self,
        *,
        row: dict[str, Any],
        metadata: dict[str, Any],
        evaluation: dict[str, Any],
        tags: Iterable[str] | None = None,
        reference_example_id: str | None = None,
    ) -> str | None:
        sample_key = sample_key_for_row(row)
        root = self.create_root(
            name="symbolic_moe_evaluation",
            inputs=root_trace_inputs(row),
            metadata={
                **metadata,
                "sample_id": row.get("id"),
                "sample_key": sample_key,
                "comparison_group": sample_key,
            },
            tags=tags,
            reference_example_id=reference_example_id,
        )
        self.create_child(
            root,
            name="evaluation",
            inputs={"predictions": row.get("predictions", [])},
            outputs=evaluation,
            metadata={"stage": "evaluation"},
        )
        self.end_run(root, outputs={"evaluation": evaluation})
        return str(root.id) if root is not None else None

    def ensure_dataset(
        self,
        *,
        dataset_name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        inputs_schema: dict[str, Any] | None = None,
        outputs_schema: dict[str, Any] | None = None,
    ):
        if not self.can_trace or self.client is None:
            return None
        if self.client.has_dataset(dataset_name=dataset_name):
            return self.client.read_dataset(dataset_name=dataset_name)
        return self.client.create_dataset(
            dataset_name=dataset_name,
            description=description,
            metadata=_safe_json(metadata or {}),
            inputs_schema=_safe_json(inputs_schema) if inputs_schema else None,
            outputs_schema=_safe_json(outputs_schema) if outputs_schema else None,
        )

    def dataset_example_map(self, *, dataset_name: str) -> dict[str, str]:
        if not self.can_trace or self.client is None:
            return {}
        cached = self._dataset_example_cache.get(dataset_name)
        if cached is not None:
            return cached
        dataset = self.ensure_dataset(dataset_name=dataset_name)
        if dataset is None:
            return {}
        mapping: dict[str, str] = {}
        for example in self.client.list_examples(dataset_id=dataset.id):
            metadata = example.metadata if isinstance(example.metadata, dict) else {}
            sample_key = metadata.get("sample_key") or metadata.get("sample_id")
            if sample_key is None:
                continue
            mapping[str(sample_key)] = str(example.id)
        self._dataset_example_cache[dataset_name] = mapping
        return mapping

    def upsert_kv_examples(
        self,
        *,
        dataset_name: str,
        rows: Sequence[dict[str, Any]],
        reference_outputs: Sequence[dict[str, Any]],
        metadata_rows: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        if not self.can_trace or self.client is None or not rows:
            return
        dataset = self.ensure_dataset(dataset_name=dataset_name)
        if dataset is None:
            return
        existing = {
            str(example.metadata.get("sample_key") or example.metadata.get("sample_id")): example
            for example in self.client.list_examples(dataset_id=dataset.id)
            if isinstance(example.metadata, dict) and example.metadata.get("sample_id") is not None
        }
        new_examples = []
        for index, row in enumerate(rows):
            sample_key = str(
                (
                    metadata_rows[index].get("sample_key")
                    if metadata_rows is not None and isinstance(metadata_rows[index], dict)
                    else f"{row.get('dataset')}::{row.get('id')}"
                )
            )
            if sample_key in existing:
                continue
            meta = dict(metadata_rows[index] if metadata_rows is not None else {})
            meta.setdefault("sample_id", row.get("id"))
            meta.setdefault("dataset", row.get("dataset"))
            meta.setdefault("sample_key", sample_key)
            new_examples.append(
                {
                    "inputs": _safe_json(row),
                    "outputs": _safe_json(reference_outputs[index]),
                    "metadata": _safe_json(meta),
                }
            )
        if new_examples:
            self.client.create_examples(dataset_id=dataset.id, examples=new_examples)

    def ensure_annotation_queue(self, *, name: str, description: str | None = None):
        if not self.can_trace or self.client is None:
            return None
        queues = list(self.client.list_annotation_queues(name=name, limit=1))
        if queues:
            return queues[0]
        return self.client.create_annotation_queue(name=name, description=description)

    def add_runs_to_queue(self, *, queue_name: str, run_ids: Sequence[str], description: str | None = None) -> None:
        if not self.can_trace or self.client is None or not run_ids:
            return
        queue = self.ensure_annotation_queue(name=queue_name, description=description)
        if queue is None:
            return
        self.client.add_runs_to_annotation_queue(queue.id, run_ids=list(run_ids))


def build_common_metadata(
    *,
    skills_path: Path | None = None,
    profiles_path: Path | None = None,
    split: str | None = None,
    extra: dict[str, Any] | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    metadata = {
        "split": split,
        "skills_hash": file_sha256(skills_path),
        "profiles_hash": file_sha256(profiles_path),
        "git_commit": git_commit(cwd),
    }
    if extra:
        metadata.update(extra)
    return {k: _safe_json(v) for k, v in metadata.items() if v is not None}
