#!/usr/bin/env python3
"""Evaluate HateBench routed outputs with binary hate metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_binary_label(text: str | None) -> str | None:
    if not text:
        return None
    t = " ".join(str(text).strip().lower().split())
    if t in {"hate", "hatespeech", "hate speech"}:
        return "hate"
    if t in {"not hate", "non-hate", "non hate", "normal"}:
        return "not hate"
    return None


def pick_pred_label(row: dict[str, Any], mode: str, expert_name: str) -> str | None:
    preds = row.get("predictions")
    if not isinstance(preds, list):
        return None
    pred_objs = [p for p in preds if isinstance(p, dict)]
    if not pred_objs:
        return None

    if mode == "expert":
        for p in pred_objs:
            if p.get("expert") == expert_name:
                return parse_binary_label(p.get("normalized_prediction") or p.get("raw_prediction"))
        return None

    if mode == "top_weight":
        best = max(pred_objs, key=lambda p: float(p.get("weight", float("-inf"))))
        return parse_binary_label(best.get("normalized_prediction") or best.get("raw_prediction"))

    if mode == "any_hate":
        labels = [
            parse_binary_label(p.get("normalized_prediction") or p.get("raw_prediction"))
            for p in pred_objs
        ]
        if "hate" in labels:
            return "hate"
        if "not hate" in labels:
            return "not hate"
        return None

    raise ValueError(f"Unsupported mode: {mode}")


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("hatebench/hatebench_smoke_outputs.jsonl"),
        help="JSONL routed outputs for HateBench.",
    )
    parser.add_argument(
        "--mode",
        choices=("expert", "top_weight", "any_hate"),
        default="expert",
        help="How to derive the final binary prediction from routed experts.",
    )
    parser.add_argument(
        "--expert-name",
        default="dynahate_hate",
        help="Expert to use in --mode expert.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("hatebench/hatebench_smoke_binary_metrics.txt"),
        help="Where to save text metrics.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)

    tp = fp = tn = fn = 0
    used = 0
    skipped = 0

    for row in rows:
        gold = parse_binary_label(row.get("output"))
        pred = pick_pred_label(row, args.mode, args.expert_name)
        if gold is None or pred is None:
            skipped += 1
            continue
        used += 1
        if gold == "hate" and pred == "hate":
            tp += 1
        elif gold == "not hate" and pred == "hate":
            fp += 1
        elif gold == "not hate" and pred == "not hate":
            tn += 1
        elif gold == "hate" and pred == "not hate":
            fn += 1

    accuracy = safe_div(tp + tn, used)
    precision_hate = safe_div(tp, tp + fp)
    recall_hate = safe_div(tp, tp + fn)
    f1_hate = safe_div(2 * precision_hate * recall_hate, precision_hate + recall_hate)

    precision_not_hate = safe_div(tn, tn + fn)
    recall_not_hate = safe_div(tn, tn + fp)
    f1_not_hate = safe_div(
        2 * precision_not_hate * recall_not_hate,
        precision_not_hate + recall_not_hate,
    )
    macro_f1 = (f1_hate + f1_not_hate) / 2

    lines = [
        f"Input: {args.input}",
        f"Mode: {args.mode}" + (f" (expert={args.expert_name})" if args.mode == "expert" else ""),
        f"Used rows: {used}",
        f"Skipped rows: {skipped}",
        f"Accuracy: {accuracy:.4f}",
        f"Precision (hate): {precision_hate:.4f}",
        f"Recall (hate): {recall_hate:.4f}",
        f"F1 (hate): {f1_hate:.4f}",
        f"Precision (not hate): {precision_not_hate:.4f}",
        f"Recall (not hate): {recall_not_hate:.4f}",
        f"F1 (not hate): {f1_not_hate:.4f}",
        f"Macro F1: {macro_f1:.4f}",
        f"Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}",
    ]

    print("\n".join(lines))
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
