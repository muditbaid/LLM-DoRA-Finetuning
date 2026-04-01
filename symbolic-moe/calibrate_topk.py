#!/usr/bin/env python3
"""Calibrate NB top-k on validation metrics using explicit routing-oriented criteria."""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


OVERALL_RE = re.compile(r"^Overall:\s+\d+/\d+\s+correct\s+\(([\d.]+)\s+accuracy\)")
ROUTING_RE = re.compile(r"^Routing precision \(top-1\):\s+\d+/\d+\s+\(([\d.]+)\)")
GOLD_TOPK_RE = re.compile(r"^Gold expert recall \(top-(\d)\):\s+\d+/\d+\s+\(([\d.]+)\)")
MACRO_RE = re.compile(r"^Overall F1 \(macro over experts\):\s+([\d.]+)")
MICRO_RE = re.compile(r"^Overall F1 \(micro over experts\):\s+([\d.]+)")


@dataclass
class RunMetrics:
    k: int
    path: Path
    overall_acc: float | None = None
    routing_top1: float | None = None
    gold_top1: float | None = None
    gold_top2: float | None = None
    gold_top3: float | None = None
    gold_top4: float | None = None
    macro_f1: float | None = None
    micro_f1: float | None = None

    @property
    def gold_at_k(self) -> float | None:
        return getattr(self, f"gold_top{self.k}", None)


def parse_metrics(path: Path, k: int) -> RunMetrics:
    metrics = RunMetrics(k=k, path=path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if m := OVERALL_RE.match(line):
            metrics.overall_acc = float(m.group(1))
        elif m := ROUTING_RE.match(line):
            metrics.routing_top1 = float(m.group(1))
        elif m := GOLD_TOPK_RE.match(line):
            setattr(metrics, f"gold_top{m.group(1)}", float(m.group(2)))
        elif m := MACRO_RE.match(line):
            metrics.macro_f1 = float(m.group(1))
        elif m := MICRO_RE.match(line):
            metrics.micro_f1 = float(m.group(1))
    return metrics


def fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "--"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def select_k(
    runs: list[RunMetrics],
    min_gold_recall: float,
    max_routing_drop: float,
) -> tuple[RunMetrics, list[RunMetrics], float]:
    best_routing = max((run.routing_top1 for run in runs if run.routing_top1 is not None), default=0.0)
    eligible: list[RunMetrics] = []
    for run in runs:
        gold_k = run.gold_at_k
        routing = run.routing_top1
        if gold_k is None or routing is None:
            continue
        if gold_k < min_gold_recall:
            continue
        if best_routing - routing > max_routing_drop:
            continue
        eligible.append(run)
    chosen_pool = eligible if eligible else [run for run in runs if run.gold_at_k is not None]
    chosen = min(
        chosen_pool,
        key=lambda run: (
            run.k,
            -(run.gold_at_k or 0.0),
            -(run.routing_top1 or 0.0),
        ),
    )
    return chosen, eligible, best_routing


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate top-k using validation routing-oriented criteria.")
    parser.add_argument(
        "--top1",
        type=Path,
        default=Path(__file__).resolve().parent / "validation_metrics_nb_top1_from_top2.txt",
        help="Validation metrics file for NB top-1.",
    )
    parser.add_argument(
        "--top2",
        type=Path,
        default=Path(__file__).resolve().parent / "validation_metrics_nb_top2.txt",
        help="Validation metrics file for NB top-2.",
    )
    parser.add_argument(
        "--top3",
        type=Path,
        default=Path(__file__).resolve().parent / "validation_metrics_nb_top3.txt",
        help="Validation metrics file for NB top-3.",
    )
    parser.add_argument(
        "--min-gold-recall",
        type=float,
        default=0.70,
        help="Minimum acceptable gold-expert recall@k on validation.",
    )
    parser.add_argument(
        "--max-routing-drop",
        type=float,
        default=0.02,
        help="Maximum allowed drop from the best validation routing top-1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "nb_topk_calibration.md",
        help="Markdown report path.",
    )
    args = parser.parse_args()

    runs = [
        parse_metrics(args.top1, 1),
        parse_metrics(args.top2, 2),
        parse_metrics(args.top3, 3),
    ]
    chosen, eligible, best_routing = select_k(
        runs=runs,
        min_gold_recall=args.min_gold_recall,
        max_routing_drop=args.max_routing_drop,
    )

    rows = [
        [
            f"Top-{run.k}",
            fmt(run.gold_at_k),
            fmt(run.routing_top1),
            fmt(run.overall_acc),
            fmt(run.macro_f1),
            fmt(run.micro_f1),
            "yes" if run in eligible else "no",
        ]
        for run in runs
    ]

    rationale_lines = [
        "# NB Top-k Calibration",
        "",
        "Validation-only calibration of the Bernoulli router execution budget.",
        "",
        "## Selection Rule",
        "",
        f"- Minimum gold-expert recall@k: `{args.min_gold_recall:.2f}`",
        f"- Maximum routing top-1 drop from the best candidate: `{args.max_routing_drop:.2f}`",
        "- Among eligible candidates, choose the smallest `k`.",
        "- If no candidate is eligible, fall back to the smallest `k` with available gold-expert recall.",
        "",
        "## Candidate Summary",
        "",
        render_table(
            ["Run", "Gold Recall@k", "Routing Top-1", "Overall Acc", "Macro F1", "Micro F1", "Eligible"],
            rows,
        ),
        "",
        "## Decision",
        "",
        f"- Best validation routing top-1: `{best_routing:.4f}`",
        f"- Selected execution budget: `top-{chosen.k}`",
        f"- Selected validation gold-expert recall@k: `{fmt(chosen.gold_at_k)}`",
        f"- Selected validation routing top-1: `{fmt(chosen.routing_top1)}`",
        f"- Selected validation overall accuracy: `{fmt(chosen.overall_acc)}`",
        "",
        "## Interpretation",
        "",
        "This calibration prioritizes routing quality over permissive end-to-end accuracy alone. "
        "Overall accuracy and F1 are retained as secondary indicators of deployment behavior rather "
        "than the primary selection rule.",
        "",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(rationale_lines), encoding="utf-8")
    print(f"Selected top-{chosen.k}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
