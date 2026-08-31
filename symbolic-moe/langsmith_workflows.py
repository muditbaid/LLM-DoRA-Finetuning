#!/usr/bin/env python3
"""
Operational LangSmith helpers for symbolic-moe datasets and review queues.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

SYMBOLIC_ROOT = Path(__file__).resolve().parent
if str(SYMBOLIC_ROOT) not in sys.path:
    sys.path.insert(0, str(SYMBOLIC_ROOT))

from langsmith_utils import (
    LangSmithManager,
    build_common_metadata,
    default_dataset_name_for_split,
    infer_split,
    parse_tags,
    sample_key_for_row,
    task_label_space,
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


config_mod = _load_module(SYMBOLIC_ROOT / "config.py", "symbolic_moe_config_langsmith_ops")
io_mod = _load_module(SYMBOLIC_ROOT / "io_utils.py", "symbolic_moe_io_langsmith_ops")
eval_mod = _load_module(SYMBOLIC_ROOT / "evaluate_outputs.py", "symbolic_moe_eval_langsmith_ops")

EXPERTS = config_mod.EXPERTS
SKILL_FILE = config_mod.SKILL_FILE
PROFILES_PATH = config_mod.PROFILES_PATH
SKILL_FIELD = config_mod.SKILL_FIELD
read_jsonl = io_mod.read_jsonl
evaluate_row = eval_mod.evaluate_row


def _gold_expert_for_dataset(dataset: str) -> str | None:
    for cfg in EXPERTS:
        if cfg.dataset == dataset:
            return cfg.name
    return None


def _default_dataset_name(path: Path) -> str:
    split = infer_split(path)
    return default_dataset_name_for_split(split) or f"symbolic-moe-{path.stem}"


def _failure_tags(evaluation: dict, low_margin_threshold: float) -> list[str]:
    tags: list[str] = []
    if evaluation.get("empty_skill_case"):
        tags.append("empty_skill_case")
    if evaluation.get("permissive_correct") is False:
        tags.append("permissive_incorrect")
    if evaluation.get("top1_correct") is False:
        tags.append("top1_incorrect")
    if evaluation.get("gold_expert_hit_at_1") is False:
        tags.append("gold_expert_miss_at_1")
    if evaluation.get("gold_expert_hit_at_2") is False:
        tags.append("gold_expert_miss_at_2")
    margin = evaluation.get("routing_margin")
    if margin is not None and float(margin) < low_margin_threshold:
        tags.append("low_margin")
    if evaluation.get("expert_disagreement"):
        tags.append("expert_disagreement")
    if evaluation.get("cross_task_overlap"):
        tags.append("cross_task_overlap")
    return tags


def upload_dataset(args) -> None:
    rows = read_jsonl(args.input)
    ls_manager = LangSmithManager(
        project_name=args.langsmith_project,
        tags=["workflow:dataset_upload", *parse_tags(args.langsmith_tags)],
        allow_without_tracing=True,
    )
    if ls_manager.client is None:
        raise SystemExit("LangSmith client is unavailable. Install langsmith and set LANGSMITH_API_KEY.")

    dataset_name = args.dataset_name or _default_dataset_name(args.input)
    split = infer_split(args.input)
    ls_manager.ensure_dataset(
        dataset_name=dataset_name,
        description=args.description or f"Symbolic-MoE {split} dataset",
        metadata=build_common_metadata(
            skills_path=SKILL_FILE,
            profiles_path=PROFILES_PATH,
            split=split,
            cwd=Path.cwd(),
            extra={"source_path": str(args.input)},
        ),
        inputs_schema={
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "dataset": {"type": "string"},
                "input": {"type": "string"},
                "system": {"type": "string"},
                "instruction": {"type": "string"},
            },
        },
        outputs_schema={
            "type": "object",
            "properties": {
                "gold_label": {"type": "string"},
                "gold_expert": {"type": "string"},
            },
        },
    )

    dataset_inputs = [
        {
            "id": row.get("id"),
            "dataset": row.get("dataset"),
            "input": row.get("input", ""),
            "system": row.get("system", ""),
            "instruction": row.get("instruction", ""),
        }
        for row in rows
    ]
    dataset_outputs = [
        {
            "gold_label": row.get("output", ""),
            "gold_expert": _gold_expert_for_dataset(row.get("dataset", "")),
        }
        for row in rows
    ]
    metadata_rows = [
        {
            "sample_id": row.get("id"),
            "sample_key": f"{row.get('dataset')}::{row.get('id')}",
            "dataset": row.get("dataset"),
            "split": split,
        }
        for row in rows
    ]
    ls_manager.upsert_kv_examples(
        dataset_name=dataset_name,
        rows=dataset_inputs,
        reference_outputs=dataset_outputs,
        metadata_rows=metadata_rows,
    )
    print(f"[langsmith] Uploaded dataset {dataset_name} with {len(rows)} examples from {args.input}")


def backfill_traces(args) -> None:
    rows = read_jsonl(args.input)
    ls_manager = LangSmithManager(
        project_name=args.langsmith_project,
        tags=[f"router:{args.router_name}", "workflow:backfill", *parse_tags(args.langsmith_tags)],
        allow_without_tracing=True,
    )
    if ls_manager.client is None:
        raise SystemExit("LangSmith client is unavailable. Install langsmith and set LANGSMITH_API_KEY.")

    split = infer_split(args.input)
    dataset_name = args.dataset_name or _default_dataset_name(args.input)
    example_id_by_key = ls_manager.dataset_example_map(dataset_name=dataset_name) if dataset_name else {}
    base_metadata = build_common_metadata(
        skills_path=SKILL_FILE,
        profiles_path=PROFILES_PATH,
        split=split,
        cwd=Path.cwd(),
        extra={
            "router_name": args.router_name,
            "router_script": args.router_script,
            "backfilled_from": str(args.input),
            "selection_policy": args.selection_policy,
            "dataset_name": dataset_name,
        },
    )
    label_space_by_expert = {cfg.name: task_label_space(cfg.label_texts) for cfg in EXPERTS}

    processed = 0
    for row in rows:
        if args.limit is not None and processed >= args.limit:
            break
        raw_predictions = row.get("predictions", [])
        predictions = []
        for pred in raw_predictions if isinstance(raw_predictions, list) else []:
            if not isinstance(pred, dict):
                continue
            expert_name = pred.get("expert")
            predictions.append(
                {
                    "expert": expert_name,
                    "weight": pred.get("weight"),
                    "label_confidence": pred.get("label_confidence"),
                    "raw_prediction": pred.get("raw_prediction"),
                    "normalized_prediction": pred.get("normalized_prediction"),
                    "task_label_space": label_space_by_expert.get(expert_name, []),
                    "latency_seconds": None,
                }
            )

        ranked_predictions = sorted(
            [p for p in predictions if isinstance(p.get("expert"), str)],
            key=lambda p: p.get("weight", float("-inf")),
            reverse=True,
        )
        evaluation = evaluate_row(row, max_k_considered=4)
        ls_manager.record_pipeline_example(
            row=row,
            metadata={
                **base_metadata,
                "dataset": row.get("dataset"),
            },
            skill_info={
                "predicted_skills": row.get(SKILL_FIELD, []),
                "skill_vote_counts": row.get("skill_vote_counts", {}),
                "keyword_responses": row.get("keyword_responses", []),
                "unselected_skills": row.get("unselected_skills", []),
                "empty_skills": not bool(row.get(SKILL_FIELD)),
            },
            router_info={
                "score_map": {},
                "ranked_experts": [p["expert"] for p in ranked_predictions],
                "selected_experts": [p["expert"] for p in ranked_predictions],
                "routing_margin": evaluation.get("routing_margin"),
                "selection_policy": args.selection_policy,
            },
            predictions=predictions,
            evaluation=evaluation,
            reference_example_id=example_id_by_key.get(sample_key_for_row(row)),
        )
        processed += 1
    print(f"[langsmith] Backfilled {processed} traces from {args.input} into project {ls_manager.project_name}")


def summarize_hard_cases(args) -> None:
    baseline_rows = {sample_key_for_row(row): row for row in read_jsonl(args.baseline_input)}
    compare_rows = (
        {sample_key_for_row(row): row for row in read_jsonl(args.compare_input)}
        if args.compare_input is not None
        else {}
    )

    keys = sorted(baseline_rows.keys())
    dataset_summary: dict[str, dict[str, int]] = {}
    failure_summary: dict[str, int] = {}
    comparison_summary: dict[str, int] = {
        "rows": 0,
        "both_correct": 0,
        "both_incorrect": 0,
        "bernoulli_improves": 0,
        "bernoulli_regresses": 0,
        "baseline_only": 0,
        "compare_only": 0,
    }
    lines: list[str] = []

    for key in keys:
        row = baseline_rows[key]
        dataset = str(row.get("dataset", "unknown"))
        dataset_bucket = dataset_summary.setdefault(dataset, {"rows": 0})
        dataset_bucket["rows"] += 1

        evaluation = evaluate_row(row, max_k_considered=4)
        for tag in _failure_tags(evaluation, args.low_margin_threshold):
            failure_summary[tag] = failure_summary.get(tag, 0) + 1
            dataset_bucket[tag] = dataset_bucket.get(tag, 0) + 1

        if not compare_rows:
            continue

        other = compare_rows.get(key)
        if other is None:
            comparison_summary["baseline_only"] += 1
            continue

        comparison_summary["rows"] += 1
        base_correct = bool(evaluation.get("permissive_correct"))
        other_eval = evaluate_row(other, max_k_considered=4)
        other_correct = bool(other_eval.get("permissive_correct"))
        if base_correct and other_correct:
            comparison_summary["both_correct"] += 1
        elif (not base_correct) and (not other_correct):
            comparison_summary["both_incorrect"] += 1
        elif (not base_correct) and other_correct:
            comparison_summary["bernoulli_improves"] += 1
        elif base_correct and (not other_correct):
            comparison_summary["bernoulli_regresses"] += 1

    if compare_rows:
        for key in compare_rows:
            if key not in baseline_rows:
                comparison_summary["compare_only"] += 1

    lines.append(f"Baseline file: {args.baseline_input}")
    if args.compare_input is not None:
        lines.append(f"Compare file: {args.compare_input}")
    lines.append(f"Low-margin threshold: {args.low_margin_threshold}")
    lines.append("")
    lines.append("Failure Summary")
    for tag, count in sorted(failure_summary.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {tag}: {count}")
    lines.append("")
    lines.append("Per-Dataset Summary")
    for dataset, stats in sorted(dataset_summary.items()):
        lines.append(f"- {dataset}: {stats}")
    if compare_rows:
        lines.append("")
        lines.append("Baseline vs Compare")
        for key, value in comparison_summary.items():
            lines.append(f"- {key}: {value}")

    text = "\n".join(lines)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"[langsmith] Wrote hard-case summary to {args.out}")


def queue_review(args) -> None:
    ls_manager = LangSmithManager(
        project_name=args.langsmith_project,
        tags=["workflow:review_queue", *parse_tags(args.langsmith_tags)],
        allow_without_tracing=True,
    )
    if ls_manager.client is None:
        raise SystemExit("LangSmith client is unavailable. Install langsmith and set LANGSMITH_API_KEY.")

    project_names = [part.strip() for part in args.projects.split(",") if part.strip()]
    if not project_names:
        raise SystemExit("At least one project name is required.")

    run_ids: list[str] = []
    total_scanned = 0
    for run in ls_manager.client.list_runs(project_name=project_names, is_root=True):
        total_scanned += 1
        outputs = run.outputs or {}
        evaluation = outputs.get("evaluation", {}) if isinstance(outputs, dict) else {}
        if not isinstance(evaluation, dict):
            continue

        should_queue = False
        if evaluation.get("empty_skill_case"):
            should_queue = True
        if evaluation.get("permissive_correct") is False:
            should_queue = True
        if evaluation.get("top1_correct") is False:
            should_queue = True
        if evaluation.get("gold_expert_hit_at_1") is False:
            should_queue = True
        margin = evaluation.get("routing_margin")
        if margin is not None and float(margin) < args.low_margin_threshold:
            should_queue = True
        if evaluation.get("expert_disagreement"):
            should_queue = True
        if evaluation.get("cross_task_overlap") and args.include_cross_task_overlap:
            should_queue = True

        if should_queue:
            run_ids.append(str(run.id))
        if args.limit and len(run_ids) >= args.limit:
            break

    ls_manager.add_runs_to_queue(
        queue_name=args.queue_name,
        run_ids=run_ids,
        description=args.description or "Symbolic-MoE hard-case review queue",
    )
    print(
        f"[langsmith] Enqueued {len(run_ids)} runs into {args.queue_name} "
        f"after scanning {total_scanned} root runs."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LangSmith workflows for symbolic-moe.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser("upload-dataset", help="Upload a validation/test pool as a LangSmith dataset.")
    upload_parser.add_argument("--input", type=Path, required=True, help="JSONL pool file to upload.")
    upload_parser.add_argument("--dataset-name", type=str, default=None, help="Optional LangSmith dataset name.")
    upload_parser.add_argument("--description", type=str, default=None, help="Optional dataset description.")
    upload_parser.add_argument("--langsmith-project", type=str, default=None, help="Optional LangSmith project name.")
    upload_parser.add_argument("--langsmith-tags", type=str, default="", help="Optional comma-separated tags.")
    upload_parser.set_defaults(func=upload_dataset)

    backfill_parser = subparsers.add_parser(
        "backfill-traces",
        help="Backfill LangSmith traces from an existing routed output JSONL.",
    )
    backfill_parser.add_argument("--input", type=Path, required=True, help="Routed output JSONL to backfill.")
    backfill_parser.add_argument("--router-name", type=str, required=True, help="Router label, e.g. profile_logodds or skill_nb.")
    backfill_parser.add_argument("--router-script", type=str, required=True, help="Source router script label.")
    backfill_parser.add_argument(
        "--selection-policy",
        type=str,
        default="backfilled_from_outputs",
        help="Selection policy label for backfilled traces.",
    )
    backfill_parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional LangSmith dataset name used to attach reference example ids.",
    )
    backfill_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of rows to backfill.",
    )
    backfill_parser.add_argument("--langsmith-project", type=str, required=True, help="Target LangSmith project name.")
    backfill_parser.add_argument("--langsmith-tags", type=str, default="", help="Optional comma-separated tags.")
    backfill_parser.set_defaults(func=backfill_traces)

    review_parser = subparsers.add_parser("queue-review", help="Populate a LangSmith annotation queue from traced runs.")
    review_parser.add_argument(
        "--projects",
        type=str,
        required=True,
        help="Comma-separated LangSmith project names to scan, e.g. validation-profile_logodds,validation-skill_nb",
    )
    review_parser.add_argument("--queue-name", type=str, required=True, help="Annotation queue name.")
    review_parser.add_argument(
        "--low-margin-threshold",
        type=float,
        default=0.25,
        help="Queue runs whose routing margin is below this threshold.",
    )
    review_parser.add_argument(
        "--include-cross-task-overlap",
        action="store_true",
        help="Also queue runs when more than one expert fired on the same post.",
    )
    review_parser.add_argument("--limit", type=int, default=None, help="Optional max number of runs to enqueue.")
    review_parser.add_argument("--description", type=str, default=None, help="Optional queue description.")
    review_parser.add_argument("--langsmith-project", type=str, default=None, help="Optional LangSmith project name.")
    review_parser.add_argument("--langsmith-tags", type=str, default="", help="Optional comma-separated tags.")
    review_parser.set_defaults(func=queue_review)

    summary_parser = subparsers.add_parser(
        "summarize-hard-cases",
        help="Summarize hard-case categories from local routed output files.",
    )
    summary_parser.add_argument("--baseline-input", type=Path, required=True, help="Baseline routed output JSONL.")
    summary_parser.add_argument(
        "--compare-input",
        type=Path,
        default=None,
        help="Optional comparison routed output JSONL, e.g. Bernoulli outputs.",
    )
    summary_parser.add_argument(
        "--low-margin-threshold",
        type=float,
        default=0.25,
        help="Count examples below this routing margin as low-margin hard cases.",
    )
    summary_parser.add_argument("--out", type=Path, default=None, help="Optional path for the text summary.")
    summary_parser.set_defaults(func=summarize_hard_cases)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
