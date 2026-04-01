#!/usr/bin/env python3
"""Render compact Markdown and LaTeX tables from evaluation metric text files."""
from __future__ import annotations

import argparse
import re
from pathlib import Path


OVERALL_RE = re.compile(r"^Overall:\s+\d+/\d+\s+correct\s+\(([\d.]+)\s+accuracy\)")
ROUTING_RE = re.compile(r"^Routing precision \(top-1\):\s+\d+/\d+\s+\(([\d.]+)\)")
GOLD_TOPK_RE = re.compile(r"^Gold expert recall \(top-(\d)\):\s+\d+/\d+\s+\(([\d.]+)\)")
DATASET_RE = re.compile(r"^(dynahate|jigsaw_threat|kaggle_cyberbullying|tweeteval_offensive):\s+\d+/\d+\s+\(([\d.]+)\)")
EXPERT_RE = re.compile(r"^(dynahate_hate|tweeteval_offense|kaggle_bully|jigsaw_threat) \(expert\): acc \d+/\d+ \(([\d.]+)\), f1 ([\d.]+)")
MACRO_RE = re.compile(r"^Overall F1 \(macro over experts\):\s+([\d.]+)")
MICRO_RE = re.compile(r"^Overall F1 \(micro over experts\):\s+([\d.]+)")

DATASET_LABELS = {
    "dynahate": "DynaHate",
    "jigsaw_threat": "Jigsaw Threat",
    "kaggle_cyberbullying": "SOSNet Cyberbullying",
    "tweeteval_offensive": "TweetEval Offensive",
}

EXPERT_LABELS = {
    "dynahate_hate": "DynaHate Expert",
    "tweeteval_offense": "Offense Expert",
    "kaggle_bully": "Cyberbullying Expert",
    "jigsaw_threat": "Threat Expert",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Baseline metrics file.",
    )
    parser.add_argument(
        "--nb-top1",
        type=Path,
        default=None,
        help="NB top-1 metrics file.",
    )
    parser.add_argument(
        "--nb-top2",
        type=Path,
        default=None,
        help="NB top-2 metrics file.",
    )
    parser.add_argument(
        "--nb-top3",
        type=Path,
        default=None,
        help="NB top-3 metrics file.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        required=True,
        help="Markdown output path.",
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        required=True,
        help="LaTeX output path.",
    )
    return parser.parse_args()


def parse_metrics(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    out = {
        "overall_acc": None,
        "routing_top1": None,
        "gold_top1": None,
        "gold_top2": None,
        "gold_top3": None,
        "gold_top4": None,
        "macro_f1": None,
        "micro_f1": None,
        "datasets": {},
        "experts": {},
    }
    for line in lines:
        if m := OVERALL_RE.match(line):
            out["overall_acc"] = float(m.group(1))
        elif m := ROUTING_RE.match(line):
            out["routing_top1"] = float(m.group(1))
        elif m := GOLD_TOPK_RE.match(line):
            out[f"gold_top{m.group(1)}"] = float(m.group(2))
        elif m := DATASET_RE.match(line):
            out["datasets"][m.group(1)] = float(m.group(2))
        elif m := EXPERT_RE.match(line):
            out["experts"][m.group(1)] = {"acc": float(m.group(2)), "f1": float(m.group(3))}
        elif m := MACRO_RE.match(line):
            out["macro_f1"] = float(m.group(1))
        elif m := MICRO_RE.match(line):
            out["micro_f1"] = float(m.group(1))
    return out


def fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "--"


def escape_tex(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def build_runs(args: argparse.Namespace) -> list[tuple[str, dict]]:
    runs: list[tuple[str, dict]] = [("Baseline", parse_metrics(args.baseline))]
    if args.nb_top1 and args.nb_top1.exists():
        runs.append(("NB Top-1", parse_metrics(args.nb_top1)))
    if args.nb_top2 and args.nb_top2.exists():
        runs.append(("NB Top-2", parse_metrics(args.nb_top2)))
    if args.nb_top3 and args.nb_top3.exists():
        runs.append(("NB Top-3", parse_metrics(args.nb_top3)))
    return runs


def render_markdown(runs: list[tuple[str, dict]]) -> str:
    headers = ["Setting", "Overall Acc", "Routing Top-1", "Gold Top-1", "Gold Top-2", "Gold Top-3", "Macro F1", "Micro F1"]
    lines = ["# Validation Metrics Tables", "", "## Overall Comparison", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for label, metrics in runs:
        lines.append(
            "| " + " | ".join(
                [
                    label,
                    fmt(metrics["overall_acc"]),
                    fmt(metrics["routing_top1"]),
                    fmt(metrics["gold_top1"]),
                    fmt(metrics["gold_top2"]),
                    fmt(metrics["gold_top3"]),
                    fmt(metrics["macro_f1"]),
                    fmt(metrics["micro_f1"]),
                ]
            ) + " |"
        )

    lines.extend(["", "## Dataset Accuracy", ""])
    headers = ["Dataset"] + [label for label, _ in runs]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for dataset_key, dataset_label in DATASET_LABELS.items():
        row = [dataset_label]
        for _, metrics in runs:
            row.append(fmt(metrics["datasets"].get(dataset_key)))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Expert Metrics", ""])
    headers = ["Expert", "Setting", "Accuracy", "F1"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for expert_key, expert_label in EXPERT_LABELS.items():
        for run_label, metrics in runs:
            expert = metrics["experts"].get(expert_key, {})
            lines.append(
                "| " + " | ".join([expert_label, run_label, fmt(expert.get("acc")), fmt(expert.get("f1"))]) + " |"
            )

    return "\n".join(lines) + "\n"


def render_tex(runs: list[tuple[str, dict]]) -> str:
    lines = [
        "% Auto-generated by symbolic-moe/render_metrics_tables.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Validation overall comparison across routing settings.}",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Setting & Overall Acc & Routing Top-1 & Gold Top-1 & Gold Top-2 & Gold Top-3 & Macro F1 & Micro F1 \\\\",
        "\\midrule",
    ]
    for label, metrics in runs:
        lines.append(
            f"{escape_tex(label)} & {fmt(metrics['overall_acc'])} & {fmt(metrics['routing_top1'])} & "
            f"{fmt(metrics['gold_top1'])} & {fmt(metrics['gold_top2'])} & {fmt(metrics['gold_top3'])} & "
            f"{fmt(metrics['macro_f1'])} & {fmt(metrics['micro_f1'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    lines.extend(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Validation dataset accuracy by routing setting.}",
            "\\begin{tabular}{l" + "c" * len(runs) + "}",
            "\\toprule",
            "Dataset & " + " & ".join(escape_tex(label) for label, _ in runs) + " \\\\",
            "\\midrule",
        ]
    )
    for dataset_key, dataset_label in DATASET_LABELS.items():
        vals = " & ".join(fmt(metrics["datasets"].get(dataset_key)) for _, metrics in runs)
        lines.append(f"{escape_tex(dataset_label)} & {vals} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    lines.extend(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{In-domain expert accuracy and F1 by routing setting.}",
            "\\begin{tabular}{llcc}",
            "\\toprule",
            "Expert & Setting & Accuracy & F1 \\\\",
            "\\midrule",
        ]
    )
    for expert_key, expert_label in EXPERT_LABELS.items():
        for run_label, metrics in runs:
            expert = metrics["experts"].get(expert_key, {})
            lines.append(
                f"{escape_tex(expert_label)} & {escape_tex(run_label)} & {fmt(expert.get('acc'))} & {fmt(expert.get('f1'))} \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    runs = build_runs(args)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(runs), encoding="utf-8")
    args.out_tex.write_text(render_tex(runs), encoding="utf-8")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_tex}")


if __name__ == "__main__":
    main()
