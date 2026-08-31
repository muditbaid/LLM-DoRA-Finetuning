#!/usr/bin/env python3
"""
Quick evaluation script for Symbolic-MoE outputs.

Scoring rule:
- If the gold label (from `output`) appears in any expert's normalized_prediction, mark correct.
- If the gold label is a negation (not_*), also mark correct when the corresponding positive label
  is absent from all predictions (even if the negated label itself was not predicted).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, List

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


build_profiles_mod = _load_module(SYMBOLIC_ROOT / "build_profiles.py", "symbolic_moe_build_profiles_eval")
config_mod = _load_module(SYMBOLIC_ROOT / "config.py", "symbolic_moe_config_eval")
io_mod = _load_module(SYMBOLIC_ROOT / "io_utils.py", "symbolic_moe_io_eval")

normalize_label = build_profiles_mod.normalize_label
EXPERTS = config_mod.EXPERTS
read_jsonl = io_mod.read_jsonl


def _dataset_normalizer(dataset: str) -> str:
    for cfg in EXPERTS:
        if cfg.dataset == dataset:
            return cfg.normalizer
    return "simple"


def _base_label_if_negative(label: str) -> str | None:
    txt = label.strip().lower()
    if txt.startswith("not "):
        return txt[4:].strip()
    if txt.startswith("not_"):
        return txt[4:].lstrip("_")
    return None


def _extract_predictions(predictions: Any) -> List[str]:
    """Normalize prediction list into lowercase strings, skipping malformed entries."""
    if not isinstance(predictions, list):
        return []
    cleaned: List[str] = []
    for entry in predictions:
        if not isinstance(entry, dict):
            continue
        norm = entry.get("normalized_prediction")
        raw = entry.get("raw_prediction")
        if isinstance(norm, str) and norm.strip():
            cleaned.append(norm.strip().lower())
        elif isinstance(raw, str) and raw.strip():
            cleaned.append(raw.strip().lower())
    return cleaned


def _select_top_prediction(predictions: Any) -> List[dict]:
    """Return a single-entry list with the highest-weight prediction when available."""
    if not isinstance(predictions, list):
        return []
    candidates = [p for p in predictions if isinstance(p, dict)]
    if not candidates:
        return []
    if any("weight" in c for c in candidates):
        best = max(candidates, key=lambda c: c.get("weight", float("-inf")))
    else:
        best = candidates[0]
    return [best]


def _prediction_entries(predictions: Any) -> List[dict]:
    if not isinstance(predictions, list):
        return []
    return [p for p in predictions if isinstance(p, dict)]


def _gold_expert_for_dataset(dataset: str) -> str | None:
    """Map dataset name to its corresponding expert config name."""
    if not dataset:
        return None
    for cfg in EXPERTS:
        if cfg.dataset == dataset:
            return cfg.name
    return None


def _sorted_expert_names(predictions: Any) -> List[str]:
    """Return expert names sorted by descending routing weight (fallback: original order)."""
    if not isinstance(predictions, list):
        return []
    candidates = [p for p in predictions if isinstance(p, dict) and isinstance(p.get("expert"), str)]
    if not candidates:
        return []
    if any("weight" in c for c in candidates):
        candidates.sort(key=lambda c: c.get("weight", float("-inf")), reverse=True)
    return [c["expert"] for c in candidates]


def _safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def _routing_margin(predictions: Any) -> float | None:
    candidates = _prediction_entries(predictions)
    if len(candidates) < 2:
        return None
    if not all("weight" in candidate for candidate in candidates[:2]):
        return None
    ranked = sorted(candidates, key=lambda c: c.get("weight", float("-inf")), reverse=True)
    return float(ranked[0].get("weight", 0.0) - ranked[1].get("weight", 0.0))


def _is_negative_label(label: str) -> bool:
    return _base_label_if_negative(label) is not None


def _prediction_polarity(prediction: str) -> str:
    return "negative" if _is_negative_label(prediction) else "positive"


def _expert_binary_labels(cfg) -> tuple[str, str]:
    labels = [normalize_label(t, cfg.normalizer) for t in cfg.label_texts]
    unique_labels = list(dict.fromkeys(labels))
    if not unique_labels:
        return "positive", "negative"

    pos = next((l for l in unique_labels if not _is_negative_label(l)), unique_labels[0])
    neg = next((l for l in unique_labels if _base_label_if_negative(l) == pos), None)
    if neg is None:
        neg = f"not_{pos}" if "_" in pos else f"not {pos}"
    return pos, neg


def record_correct(output_label: str, predictions: Any, dataset: str) -> bool:
    normalizer = _dataset_normalizer(dataset)
    gold = normalize_label(output_label or "", normalizer)
    preds = _extract_predictions(predictions)

    if gold in preds:
        return True

    base = _base_label_if_negative(gold)
    if base:
        return base not in preds

    return False


def evaluate_row(row: dict[str, Any], max_k_considered: int = 4) -> dict[str, Any]:
    output_label = row.get("output", "")
    predictions = row.get("predictions", [])
    dataset = row.get("dataset", "")
    ranked_experts = _sorted_expert_names(predictions)
    gold_expert = _gold_expert_for_dataset(dataset)
    top1_preds = _select_top_prediction(predictions)
    prediction_entries = _prediction_entries(predictions)
    normalized_predictions = _extract_predictions(predictions)
    polarities = {_prediction_polarity(pred) for pred in normalized_predictions}

    topk_hits = {
        f"gold_expert_hit_at_{k}": bool(gold_expert and gold_expert in ranked_experts[:k])
        for k in range(1, max_k_considered + 1)
    }
    routing_margin = _routing_margin(predictions)
    selected_experts = [entry.get("expert") for entry in prediction_entries if isinstance(entry.get("expert"), str)]

    return {
        "permissive_correct": record_correct(output_label, predictions, dataset),
        "top1_correct": record_correct(output_label, top1_preds, dataset),
        "gold_expert": gold_expert,
        "ranked_experts": ranked_experts,
        "selected_experts": selected_experts,
        "num_selected_experts": len(selected_experts),
        "empty_skill_case": not bool(row.get("predicted_skills")),
        "routing_margin": routing_margin,
        "underrouted": bool(gold_expert and gold_expert not in selected_experts),
        "expert_disagreement": len(polarities) > 1,
        "cross_task_overlap": len(selected_experts) > 1,
        **topk_hits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate routed outputs against gold labels.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "test_pool_outputs.jsonl",
        help="JSONL with routed predictions (default: symbolic-moe/test_pool_outputs.jsonl)",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional path to save the printed metrics.",
    )
    parser.add_argument(
        "--langsmith-project",
        type=str,
        default=None,
        help="Optional LangSmith project override for evaluation traces.",
    )
    parser.add_argument(
        "--langsmith-tags",
        type=str,
        default="",
        help="Optional comma-separated LangSmith tags.",
    )
    parser.add_argument(
        "--langsmith-dataset-name",
        type=str,
        default=None,
        help="Optional LangSmith dataset name used to attach reference example ids.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    split = infer_split(args.input)
    ls_manager = LangSmithManager(
        project_name=args.langsmith_project,
        tags=[f"split:{split}", *parse_tags(args.langsmith_tags)],
    )
    ls_metadata = build_common_metadata(
        split=split,
        cwd=Path.cwd(),
        extra={"evaluation_source": str(args.input)},
    )
    dataset_name = args.langsmith_dataset_name or default_dataset_name_for_split(split)
    example_id_by_key = (
        ls_manager.dataset_example_map(dataset_name=dataset_name)
        if ls_manager.enabled and dataset_name
        else {}
    )
    total = len(rows)
    correct = 0
    top1_correct = 0
    per_dataset = defaultdict(lambda: {"correct": 0, "total": 0})
    per_dataset_top1 = defaultdict(lambda: {"correct": 0, "total": 0})
    topk_hits = defaultdict(int)  # k -> hits
    per_dataset_topk_hits = defaultdict(lambda: defaultdict(int))  # dataset -> (k -> hits)
    max_k_considered = 4
    expert_cfg_by_name = {cfg.name: cfg for cfg in EXPERTS}
    per_expert = defaultdict(lambda: {"correct": 0, "total": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0})

    for row in rows:
        output_label = row.get("output", "")
        predictions = row.get("predictions", [])
        dataset = row.get("dataset", "")
        row_eval = evaluate_row(row, max_k_considered=max_k_considered)
        ok = row_eval["permissive_correct"]
        correct += int(ok)
        ds_stats = per_dataset[dataset or "unknown"]
        ds_stats["total"] += 1
        ds_stats["correct"] += int(ok)
        top1_preds = _select_top_prediction(predictions)
        top1_ok = row_eval["top1_correct"]
        top1_correct += int(top1_ok)
        ds_top1 = per_dataset_top1[dataset or "unknown"]
        ds_top1["total"] += 1
        ds_top1["correct"] += int(top1_ok)

        gold_expert = row_eval["gold_expert"]
        ranked_experts = row_eval["ranked_experts"]
        if gold_expert and ranked_experts:
            for k in range(1, max_k_considered + 1):
                hit = gold_expert in ranked_experts[:k]
                topk_hits[k] += int(hit)
                per_dataset_topk_hits[dataset or "unknown"][k] += int(hit)

        if isinstance(predictions, list):
            for pred in predictions:
                if not isinstance(pred, dict):
                    continue
                expert_name = pred.get("expert")
                if not isinstance(expert_name, str):
                    continue
                cfg = expert_cfg_by_name.get(expert_name)
                if cfg is None or cfg.dataset != dataset:
                    continue

                gold = normalize_label(output_label or "", cfg.normalizer)
                pred_raw = pred.get("normalized_prediction") or pred.get("raw_prediction") or ""
                pred_norm = normalize_label(str(pred_raw), cfg.normalizer)
                stats = per_expert[expert_name]
                stats["total"] += 1
                stats["correct"] += int(pred_norm == gold)

                pos_label, neg_label = _expert_binary_labels(cfg)
                if gold not in {pos_label, neg_label} or pred_norm not in {pos_label, neg_label}:
                    continue
                if gold == pos_label and pred_norm == pos_label:
                    stats["tp"] += 1
                elif gold == neg_label and pred_norm == pos_label:
                    stats["fp"] += 1
                elif gold == neg_label and pred_norm == neg_label:
                    stats["tn"] += 1
                elif gold == pos_label and pred_norm == neg_label:
                    stats["fn"] += 1

        if ls_manager.enabled:
            ls_manager.record_evaluation_example(
                row=row,
                metadata={
                    **ls_metadata,
                    "dataset": dataset,
                },
                evaluation=row_eval,
                reference_example_id=example_id_by_key.get(sample_key_for_row(row)),
            )

    accuracy = correct / total if total else 0.0
    top1_accuracy = top1_correct / total if total else 0.0
    lines = []
    lines.append(f"Overall: {correct}/{total} correct ({accuracy:.4f} accuracy)")
    lines.append(f"Routing precision (top-1): {top1_correct}/{total} ({top1_accuracy:.4f})")
    for k in range(1, max_k_considered + 1):
        hits = topk_hits.get(k, 0)
        denom = total if total else 0
        rate = hits / denom if denom else 0.0
        lines.append(f"Gold expert recall (top-{k}): {hits}/{total} ({rate:.4f})")
    for ds, stats in sorted(per_dataset.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
        lines.append(f"{ds}: {stats['correct']}/{stats['total']} ({acc:.4f})")
    for ds, stats in sorted(per_dataset_top1.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
        lines.append(f"{ds} (top-1): {stats['correct']}/{stats['total']} ({acc:.4f})")
    for ds in sorted(per_dataset_topk_hits.keys()):
        for k in range(1, max_k_considered + 1):
            hits = per_dataset_topk_hits[ds].get(k, 0)
            denom = per_dataset_top1[ds]["total"]
            rate = hits / denom if denom else 0.0
            lines.append(f"{ds} (gold expert top-{k}): {hits}/{denom} ({rate:.4f})")

    expert_f1s = []
    total_tp = total_fp = total_fn = 0
    for cfg in EXPERTS:
        stats = per_expert.get(cfg.name)
        if not stats or stats["total"] == 0:
            continue
        acc = _safe_div(stats["correct"], stats["total"])
        precision = _safe_div(stats["tp"], stats["tp"] + stats["fp"])
        recall = _safe_div(stats["tp"], stats["tp"] + stats["fn"])
        f1 = _safe_div(2 * precision * recall, precision + recall)
        expert_f1s.append(f1)
        total_tp += stats["tp"]
        total_fp += stats["fp"]
        total_fn += stats["fn"]
        lines.append(
            f"{cfg.name} (expert): acc {stats['correct']}/{stats['total']} ({acc:.4f}), "
            f"f1 {f1:.4f}"
        )

    if expert_f1s:
        macro_f1 = _safe_div(sum(expert_f1s), len(expert_f1s))
        micro_precision = _safe_div(total_tp, total_tp + total_fp)
        micro_recall = _safe_div(total_tp, total_tp + total_fn)
        micro_f1 = _safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)
        lines.append(f"Overall F1 (macro over experts): {macro_f1:.4f}")
        lines.append(f"Overall F1 (micro over experts): {micro_f1:.4f}")
    else:
        lines.append("Overall F1 (macro over experts): N/A (no in-domain expert predictions)")
        lines.append("Overall F1 (micro over experts): N/A (no in-domain expert predictions)")
    print("\n".join(lines))

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
