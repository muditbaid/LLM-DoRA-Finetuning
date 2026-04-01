#!/usr/bin/env python3
"""Assemble neat router result summaries into a dedicated results folder."""
from __future__ import annotations

import re
from pathlib import Path


SYMBOLIC_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = SYMBOLIC_ROOT / "results"

OVERALL_RE = re.compile(r"^Overall:\s+\d+/\d+\s+correct\s+\(([\d.]+)\s+accuracy\)")
ROUTING_RE = re.compile(r"^Routing precision \(top-1\):\s+\d+/\d+\s+\(([\d.]+)\)")
GOLD_TOPK_RE = re.compile(r"^Gold expert recall \(top-(\d)\):\s+\d+/\d+\s+\(([\d.]+)\)")
DATASET_RE = re.compile(r"^(dynahate|jigsaw_threat|kaggle_cyberbullying|tweeteval_offensive):\s+\d+/\d+\s+\(([\d.]+)\)")
DATASET_TOP1_RE = re.compile(r"^(dynahate|jigsaw_threat|kaggle_cyberbullying|tweeteval_offensive) \(top-1\):\s+\d+/\d+\s+\(([\d.]+)\)")
DATASET_GOLD_RE = re.compile(r"^(dynahate|jigsaw_threat|kaggle_cyberbullying|tweeteval_offensive) \(gold expert top-(\d)\):\s+\d+/\d+\s+\(([\d.]+)\)")
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


def parse_metrics(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").splitlines()
    metrics = {
        "overall_acc": None,
        "routing_top1": None,
        "gold_top1": None,
        "gold_top2": None,
        "gold_top3": None,
        "gold_top4": None,
        "macro_f1": None,
        "micro_f1": None,
        "datasets": {},
        "dataset_top1": {},
        "dataset_gold": {},
        "experts": {},
    }
    for line in text:
        if m := OVERALL_RE.match(line):
            metrics["overall_acc"] = float(m.group(1))
        elif m := ROUTING_RE.match(line):
            metrics["routing_top1"] = float(m.group(1))
        elif m := GOLD_TOPK_RE.match(line):
            metrics[f"gold_top{m.group(1)}"] = float(m.group(2))
        elif m := DATASET_RE.match(line):
            metrics["datasets"][m.group(1)] = float(m.group(2))
        elif m := DATASET_TOP1_RE.match(line):
            metrics["dataset_top1"][m.group(1)] = float(m.group(2))
        elif m := DATASET_GOLD_RE.match(line):
            metrics["dataset_gold"].setdefault(m.group(1), {})[f"top{m.group(2)}"] = float(m.group(3))
        elif m := EXPERT_RE.match(line):
            metrics["experts"][m.group(1)] = {"acc": float(m.group(2)), "f1": float(m.group(3))}
        elif m := MACRO_RE.match(line):
            metrics["macro_f1"] = float(m.group(1))
        elif m := MICRO_RE.match(line):
            metrics["micro_f1"] = float(m.group(1))
    return metrics


def fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "--"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_baseline_files() -> None:
    val = parse_metrics(SYMBOLIC_ROOT / "validation_metrics_20260315_201425.txt")
    test = parse_metrics(SYMBOLIC_ROOT / "test_metrics_20260316_195955.txt")

    all_md = [
        "# Baseline Router: All Metrics",
        "",
        "## Overall",
        "",
        render_table(
            ["Split", "Overall Acc", "Routing Top-1", "Gold Top-1", "Gold Top-2", "Gold Top-3", "Macro F1", "Micro F1"],
            [
                ["Validation", fmt(val["overall_acc"]), fmt(val["routing_top1"]), fmt(val["gold_top1"]), fmt(val["gold_top2"]), fmt(val["gold_top3"]), fmt(val["macro_f1"]), fmt(val["micro_f1"])],
                ["Test", fmt(test["overall_acc"]), fmt(test["routing_top1"]), fmt(test["gold_top1"]), fmt(test["gold_top2"]), fmt(test["gold_top3"]), fmt(test["macro_f1"]), fmt(test["micro_f1"])],
            ],
        ),
        "",
        "## Dataset Accuracy",
        "",
        render_table(
            ["Dataset", "Validation", "Test"],
            [[DATASET_LABELS[k], fmt(val["datasets"].get(k)), fmt(test["datasets"].get(k))] for k in DATASET_LABELS],
        ),
        "",
        "## Expert Accuracy and F1",
        "",
        render_table(
            ["Expert", "Split", "Accuracy", "F1"],
            [
                [EXPERT_LABELS[k], split, fmt(metrics["experts"].get(k, {}).get("acc")), fmt(metrics["experts"].get(k, {}).get("f1"))]
                for split, metrics in (("Validation", val), ("Test", test))
                for k in EXPERT_LABELS
            ],
        ),
        "",
    ]
    (RESULTS_DIR / "baseline_all_metrics.md").write_text("\n".join(all_md), encoding="utf-8")

    routing_md = [
        "# Baseline Router: Routing Metrics",
        "",
        "## Overall Routing",
        "",
        render_table(
            ["Split", "Routing Top-1", "Gold Top-1", "Gold Top-2", "Gold Top-3", "Gold Top-4"],
            [
                ["Validation", fmt(val["routing_top1"]), fmt(val["gold_top1"]), fmt(val["gold_top2"]), fmt(val["gold_top3"]), fmt(val["gold_top4"])],
                ["Test", fmt(test["routing_top1"]), fmt(test["gold_top1"]), fmt(test["gold_top2"]), fmt(test["gold_top3"]), fmt(test["gold_top4"])],
            ],
        ),
        "",
        "## Per-Dataset Routing",
        "",
        render_table(
            ["Dataset", "Split", "Top-1 Routed Acc", "Gold Top-1", "Gold Top-2", "Gold Top-3", "Gold Top-4"],
            [
                [DATASET_LABELS[k], split, fmt(metrics["dataset_top1"].get(k)), fmt(metrics["dataset_gold"].get(k, {}).get("top1")), fmt(metrics["dataset_gold"].get(k, {}).get("top2")), fmt(metrics["dataset_gold"].get(k, {}).get("top3")), fmt(metrics["dataset_gold"].get(k, {}).get("top4"))]
                for split, metrics in (("Validation", val), ("Test", test))
                for k in DATASET_LABELS
            ],
        ),
        "",
    ]
    (RESULTS_DIR / "baseline_routing_metrics.md").write_text("\n".join(routing_md), encoding="utf-8")


def write_nb_files() -> None:
    val_top1 = parse_metrics(SYMBOLIC_ROOT / "validation_metrics_nb_top1_from_top2.txt")
    val_top2 = parse_metrics(SYMBOLIC_ROOT / "validation_metrics_nb_top2.txt")
    val_top3 = parse_metrics(SYMBOLIC_ROOT / "validation_metrics_nb_top3.txt")
    test_top1 = parse_metrics(SYMBOLIC_ROOT / "test_metrics_nb_top1_from_top2.txt")
    test_top2 = parse_metrics(SYMBOLIC_ROOT / "test_metrics_nb_top2.txt")
    test_top3 = parse_metrics(SYMBOLIC_ROOT / "test_metrics_nb_top3.txt")

    all_md = [
        "# Bernoulli Router: All Metrics",
        "",
        "## Overall",
        "",
        render_table(
            ["Run", "Split", "Overall Acc", "Routing Top-1", "Gold Top-1", "Gold Top-2", "Gold Top-3", "Macro F1", "Micro F1"],
            [
                ["NB Top-1", "Validation", fmt(val_top1["overall_acc"]), fmt(val_top1["routing_top1"]), fmt(val_top1["gold_top1"]), fmt(val_top1["gold_top2"]), fmt(val_top1["gold_top3"]), fmt(val_top1["macro_f1"]), fmt(val_top1["micro_f1"])],
                ["NB Top-2", "Validation", fmt(val_top2["overall_acc"]), fmt(val_top2["routing_top1"]), fmt(val_top2["gold_top1"]), fmt(val_top2["gold_top2"]), fmt(val_top2["gold_top3"]), fmt(val_top2["macro_f1"]), fmt(val_top2["micro_f1"])],
                ["NB Top-3", "Validation", fmt(val_top3["overall_acc"]), fmt(val_top3["routing_top1"]), fmt(val_top3["gold_top1"]), fmt(val_top3["gold_top2"]), fmt(val_top3["gold_top3"]), fmt(val_top3["macro_f1"]), fmt(val_top3["micro_f1"])],
                ["NB Top-1", "Test", fmt(test_top1["overall_acc"]), fmt(test_top1["routing_top1"]), fmt(test_top1["gold_top1"]), fmt(test_top1["gold_top2"]), fmt(test_top1["gold_top3"]), fmt(test_top1["macro_f1"]), fmt(test_top1["micro_f1"])],
                ["NB Top-2", "Test", fmt(test_top2["overall_acc"]), fmt(test_top2["routing_top1"]), fmt(test_top2["gold_top1"]), fmt(test_top2["gold_top2"]), fmt(test_top2["gold_top3"]), fmt(test_top2["macro_f1"]), fmt(test_top2["micro_f1"])],
                ["NB Top-3", "Test", fmt(test_top3["overall_acc"]), fmt(test_top3["routing_top1"]), fmt(test_top3["gold_top1"]), fmt(test_top3["gold_top2"]), fmt(test_top3["gold_top3"]), fmt(test_top3["macro_f1"]), fmt(test_top3["micro_f1"])],
            ],
        ),
        "",
        "## Dataset Accuracy",
        "",
        render_table(
            ["Dataset", "Val Top-1", "Val Top-2", "Val Top-3", "Test Top-1", "Test Top-2", "Test Top-3"],
            [
                [
                    DATASET_LABELS[k],
                    fmt(val_top1["datasets"].get(k)),
                    fmt(val_top2["datasets"].get(k)),
                    fmt(val_top3["datasets"].get(k)),
                    fmt(test_top1["datasets"].get(k)),
                    fmt(test_top2["datasets"].get(k)),
                    fmt(test_top3["datasets"].get(k)),
                ]
                for k in DATASET_LABELS
            ],
        ),
        "",
        "## Expert Accuracy and F1",
        "",
        render_table(
            ["Expert", "Run", "Accuracy", "F1"],
            [
                [EXPERT_LABELS[k], run, fmt(metrics["experts"].get(k, {}).get("acc")), fmt(metrics["experts"].get(k, {}).get("f1"))]
                for run, metrics in (
                    ("Val Top-1", val_top1),
                    ("Val Top-2", val_top2),
                    ("Val Top-3", val_top3),
                    ("Test Top-1", test_top1),
                    ("Test Top-2", test_top2),
                    ("Test Top-3", test_top3),
                )
                for k in EXPERT_LABELS
            ],
        ),
        "",
    ]
    (RESULTS_DIR / "nb_all_metrics.md").write_text("\n".join(all_md), encoding="utf-8")

    routing_md = [
        "# Bernoulli Router: Routing Metrics",
        "",
        "## Overall Routing",
        "",
        render_table(
            ["Run", "Split", "Routing Top-1", "Gold Top-1", "Gold Top-2", "Gold Top-3", "Gold Top-4"],
            [
                ["NB Top-1", "Validation", fmt(val_top1["routing_top1"]), fmt(val_top1["gold_top1"]), fmt(val_top1["gold_top2"]), fmt(val_top1["gold_top3"]), fmt(val_top1["gold_top4"])],
                ["NB Top-2", "Validation", fmt(val_top2["routing_top1"]), fmt(val_top2["gold_top1"]), fmt(val_top2["gold_top2"]), fmt(val_top2["gold_top3"]), fmt(val_top2["gold_top4"])],
                ["NB Top-3", "Validation", fmt(val_top3["routing_top1"]), fmt(val_top3["gold_top1"]), fmt(val_top3["gold_top2"]), fmt(val_top3["gold_top3"]), fmt(val_top3["gold_top4"])],
                ["NB Top-1", "Test", fmt(test_top1["routing_top1"]), fmt(test_top1["gold_top1"]), fmt(test_top1["gold_top2"]), fmt(test_top1["gold_top3"]), fmt(test_top1["gold_top4"])],
                ["NB Top-2", "Test", fmt(test_top2["routing_top1"]), fmt(test_top2["gold_top1"]), fmt(test_top2["gold_top2"]), fmt(test_top2["gold_top3"]), fmt(test_top2["gold_top4"])],
                ["NB Top-3", "Test", fmt(test_top3["routing_top1"]), fmt(test_top3["gold_top1"]), fmt(test_top3["gold_top2"]), fmt(test_top3["gold_top3"]), fmt(test_top3["gold_top4"])],
            ],
        ),
        "",
        "## Per-Dataset Routing",
        "",
        render_table(
            ["Dataset", "Run", "Top-1 Routed Acc", "Gold Top-1", "Gold Top-2", "Gold Top-3", "Gold Top-4"],
            [
                [DATASET_LABELS[k], run, fmt(metrics["dataset_top1"].get(k)), fmt(metrics["dataset_gold"].get(k, {}).get("top1")), fmt(metrics["dataset_gold"].get(k, {}).get("top2")), fmt(metrics["dataset_gold"].get(k, {}).get("top3")), fmt(metrics["dataset_gold"].get(k, {}).get("top4"))]
                for run, metrics in (
                    ("Val Top-1", val_top1),
                    ("Val Top-2", val_top2),
                    ("Val Top-3", val_top3),
                    ("Test Top-1", test_top1),
                    ("Test Top-2", test_top2),
                    ("Test Top-3", test_top3),
                )
                for k in DATASET_LABELS
            ],
        ),
        "",
    ]
    (RESULTS_DIR / "nb_routing_metrics.md").write_text("\n".join(routing_md), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_baseline_files()
    write_nb_files()
    print(f"Wrote results into {RESULTS_DIR}")


if __name__ == "__main__":
    main()
