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
from collections import defaultdict
from pathlib import Path
from typing import List, Any

SYMBOLIC_ROOT = Path(__file__).resolve().parent


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
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    total = len(rows)
    correct = 0
    top1_correct = 0
    per_dataset = defaultdict(lambda: {"correct": 0, "total": 0})
    per_dataset_top1 = defaultdict(lambda: {"correct": 0, "total": 0})

    for row in rows:
        output_label = row.get("output", "")
        predictions = row.get("predictions", [])
        dataset = row.get("dataset", "")
        ok = record_correct(output_label, predictions, dataset)
        correct += int(ok)
        ds_stats = per_dataset[dataset or "unknown"]
        ds_stats["total"] += 1
        ds_stats["correct"] += int(ok)
        top1_preds = _select_top_prediction(predictions)
        top1_ok = record_correct(output_label, top1_preds, dataset)
        top1_correct += int(top1_ok)
        ds_top1 = per_dataset_top1[dataset or "unknown"]
        ds_top1["total"] += 1
        ds_top1["correct"] += int(top1_ok)

    accuracy = correct / total if total else 0.0
    top1_accuracy = top1_correct / total if total else 0.0
    lines = []
    lines.append(f"Overall: {correct}/{total} correct ({accuracy:.4f} accuracy)")
    lines.append(f"Routing precision (top-1): {top1_correct}/{total} ({top1_accuracy:.4f})")
    for ds, stats in sorted(per_dataset.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
        lines.append(f"{ds}: {stats['correct']}/{stats['total']} ({acc:.4f})")
    for ds, stats in sorted(per_dataset_top1.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
        lines.append(f"{ds} (top-1): {stats['correct']}/{stats['total']} ({acc:.4f})")
    print("\n".join(lines))

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
