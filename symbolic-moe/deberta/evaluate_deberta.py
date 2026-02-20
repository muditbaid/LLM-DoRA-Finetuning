#!/usr/bin/env python3
"""
Evaluate DeBERTa multi-label predictions against gold labels.
"""
from __future__ import annotations

import argparse
import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

SYMBOLIC_ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


io_utils = _load_module(SYMBOLIC_ROOT / "io_utils.py", "symbolic_moe_io_deberta_eval")

LABELS = ["hate", "offense", "bully", "threat"]
DATASET_TO_LABEL = {
    "dynahate": "hate",
    "tweeteval_offensive": "offense",
    "kaggle_cyberbullying": "bully",
    "jigsaw_threat": "threat",
}
POSITIVE_BY_LABEL = {
    "hate": {"hate"},
    "offense": {"offensive"},
    "bully": {"bully"},
    "threat": {"threat"},
}


def _gold_is_positive(sample: Dict[str, str]) -> Optional[bool]:
    dataset = sample.get("dataset") or ""
    label_name = DATASET_TO_LABEL.get(dataset)
    if not label_name:
        return None
    raw_label = (sample.get("label") or "").strip().lower()
    return raw_label in POSITIVE_BY_LABEL[label_name]


def _predicted_labels(sample: Dict[str, object]) -> List[str]:
    preds = sample.get("deberta_predicted", [])
    if not isinstance(preds, list):
        return []
    return [p for p in preds if isinstance(p, str)]


def record_correct(sample: Dict[str, object]) -> Optional[bool]:
    dataset = sample.get("dataset") or ""
    label_name = DATASET_TO_LABEL.get(dataset)
    if not label_name:
        return None
    gold_positive = _gold_is_positive(sample)
    if gold_positive is None:
        return None
    preds = _predicted_labels(sample)
    if gold_positive:
        return label_name in preds
    return label_name not in preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeBERTa predictions.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("symbolic-moe/deberta/pilot_predictions.jsonl"),
        help="JSONL with DeBERTa predictions.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional path to save the printed metrics.",
    )
    args = parser.parse_args()

    rows = io_utils.read_jsonl(args.input)
    total = 0
    correct = 0
    per_dataset = defaultdict(lambda: {"correct": 0, "total": 0})

    for row in tqdm(rows, desc="evaluate rows"):
        ok = record_correct(row)
        if ok is None:
            continue
        total += 1
        correct += int(ok)
        dataset = row.get("dataset") or "unknown"
        stats = per_dataset[dataset]
        stats["total"] += 1
        stats["correct"] += int(ok)

    accuracy = correct / total if total else 0.0
    lines = [f"Overall: {correct}/{total} correct ({accuracy:.4f} accuracy)"]
    for ds, stats in sorted(per_dataset.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
        lines.append(f"{ds}: {stats['correct']}/{stats['total']} ({acc:.4f})")

    print("\n".join(lines))
    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
