#!/usr/bin/env python3
"""Analyze empty-skill rows and save summary statistics, examples, and charts."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


SYMBOLIC_ROOT = Path(__file__).resolve().parent
TITLE_SIZE = 10.0
LABEL_SIZE = 9.0
TICK_SIZE = 8.5
ANNOTATION_SIZE = 8.0
BAR_EDGE_WIDTH = 0.6
GRID_WIDTH = 0.8
EXPORT_DPI = 300

plt.rcParams.update(
    {
        "font.size": TICK_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": TICK_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

DATASET_DISPLAY = {
    "dynahate": "DynaHate",
    "tweeteval_offensive": "TweetEval Offensive",
    "kaggle_cyberbullying": "SOSNet Cyberbullying",
    "jigsaw_threat": "Jigsaw Threat",
}

FEATURE_PATTERNS = {
    "very_short": None,
    "has_mention": re.compile(r"@\w+"),
    "has_hashtag": re.compile(r"#\w+"),
    "has_url": re.compile(r"https?://|www\.|t\.co/"),
    "has_non_ascii": re.compile(r"[^\x00-\x7F]"),
    "has_profanity": re.compile(r"\b(fuck|fucking|shit|bitch|asshole|dick|bastard|cunt|arse)\b", re.I),
    "has_quote": re.compile(r"[\"“”']"),
}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


io_mod = _load_module(SYMBOLIC_ROOT / "io_utils.py", "symbolic_moe_io_empty_stats")
eval_mod = _load_module(SYMBOLIC_ROOT / "evaluate_outputs.py", "symbolic_moe_eval_empty_stats")
read_jsonl = io_mod.read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skills-file",
        type=Path,
        required=True,
        help="JSONL containing predicted_skills and labels.",
    )
    parser.add_argument(
        "--outputs-file",
        type=Path,
        default=None,
        help="Optional routed outputs JSONL for routing metrics.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to save summary files and charts.",
    )
    parser.add_argument(
        "--example-count",
        type=int,
        default=8,
        help="Examples per dataset and label polarity to save.",
    )
    return parser.parse_args()


def is_positive(row: dict) -> bool:
    dataset = row.get("dataset", "")
    normalizer = eval_mod._dataset_normalizer(dataset)
    gold = eval_mod.normalize_label(row.get("output", ""), normalizer)
    return eval_mod._base_label_if_negative(gold) is None


def dataset_order(rows: list[dict]) -> list[str]:
    present = {row.get("dataset") for row in rows if isinstance(row.get("dataset"), str)}
    preferred = [d for d in DATASET_DISPLAY if d in present]
    extras = sorted(present - set(preferred))
    return preferred + extras


def safe_rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def wrap_label(text: str, width: int = 12) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False)) or text


def build_dataset_summary(rows: list[dict], outputs_by_id: dict[str, dict] | None) -> list[dict[str, Any]]:
    order = dataset_order(rows)
    summaries: list[dict[str, Any]] = []

    for dataset in order:
        ds_rows = [row for row in rows if row.get("dataset") == dataset]
        empty_rows = [row for row in ds_rows if not row.get("predicted_skills")]
        nonempty_rows = [row for row in ds_rows if row.get("predicted_skills")]

        summary: dict[str, Any] = {
            "dataset": dataset,
            "dataset_display": DATASET_DISPLAY.get(dataset, dataset),
            "n_rows": len(ds_rows),
            "empty_rows": len(empty_rows),
            "nonempty_rows": len(nonempty_rows),
            "empty_rate": safe_rate(len(empty_rows), len(ds_rows)),
            "empty_positive": sum(is_positive(row) for row in empty_rows),
            "empty_negative": sum(not is_positive(row) for row in empty_rows),
            "nonempty_positive": sum(is_positive(row) for row in nonempty_rows),
            "nonempty_negative": sum(not is_positive(row) for row in nonempty_rows),
        }
        summary["empty_positive_rate"] = safe_rate(summary["empty_positive"], summary["empty_rows"])
        summary["nonempty_positive_rate"] = safe_rate(summary["nonempty_positive"], summary["nonempty_rows"])

        if outputs_by_id is not None:
            for group_name, group_rows in (("empty", empty_rows), ("nonempty", nonempty_rows)):
                top_hits = {1: 0, 2: 0, 3: 0, 4: 0}
                correct = 0
                total = 0
                gold_expert = eval_mod._gold_expert_for_dataset(dataset)

                for row in group_rows:
                    out = outputs_by_id.get(row.get("id"))
                    if out is None:
                        continue
                    total += 1
                    ranked = eval_mod._sorted_expert_names(out.get("predictions", []))
                    for k in top_hits:
                        top_hits[k] += int(gold_expert in ranked[:k])
                    correct += int(
                        eval_mod.record_correct(out.get("output", ""), out.get("predictions", []), dataset)
                    )

                summary[f"{group_name}_routing_rows_with_outputs"] = total
                summary[f"{group_name}_gold_top1"] = safe_rate(top_hits[1], total)
                summary[f"{group_name}_gold_top2"] = safe_rate(top_hits[2], total)
                summary[f"{group_name}_gold_top3"] = safe_rate(top_hits[3], total)
                summary[f"{group_name}_gold_top4"] = safe_rate(top_hits[4], total)
                summary[f"{group_name}_permissive_acc"] = safe_rate(correct, total)

        summaries.append(summary)

    return summaries


def build_group_feature_summary(rows: list[dict]) -> list[dict[str, Any]]:
    groups = {"empty": [row for row in rows if not row.get("predicted_skills")], "nonempty": [row for row in rows if row.get("predicted_skills")]}
    summaries: list[dict[str, Any]] = []

    for group_name, group_rows in groups.items():
        counter = Counter()
        total_words = 0
        for row in group_rows:
            text = str(row.get("input", "") or "").strip()
            tokens = text.split()
            total_words += len(tokens)
            counter["positive"] += int(is_positive(row))
            counter["negative"] += int(not is_positive(row))
            counter["very_short"] += int(len(tokens) <= 6)
            for feature, pattern in FEATURE_PATTERNS.items():
                if feature == "very_short":
                    continue
                counter[feature] += int(bool(pattern.search(text)))

        n_rows = len(group_rows)
        summary: dict[str, Any] = {
            "group": group_name,
            "n_rows": n_rows,
            "avg_words": safe_rate(total_words, n_rows),
        }
        for key, value in counter.items():
            summary[key] = value
            summary[f"{key}_rate"] = safe_rate(value, n_rows)
        summaries.append(summary)

    return summaries


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_examples(path: Path, rows: list[dict], example_count: int) -> None:
    order = dataset_order(rows)
    lines: list[str] = []
    for dataset in order:
        ds_rows = [row for row in rows if row.get("dataset") == dataset and not row.get("predicted_skills")]
        positives = [row for row in ds_rows if is_positive(row)]
        negatives = [row for row in ds_rows if not is_positive(row)]

        lines.append(f"DATASET: {DATASET_DISPLAY.get(dataset, dataset)}")
        lines.append(f"EMPTY COUNT: {len(ds_rows)}")
        lines.append("POSITIVE EXAMPLES:")
        for row in positives[:example_count]:
            lines.append(f"- {row.get('id')} | {row.get('output')} | {str(row.get('input', '')).replace(chr(10), ' ')[:260]}")
        lines.append("NEGATIVE EXAMPLES:")
        for row in negatives[:example_count]:
            lines.append(f"- {row.get('id')} | {row.get('output')} | {str(row.get('input', '')).replace(chr(10), ' ')[:260]}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def save_summary_txt(path: Path, dataset_rows: list[dict[str, Any]], group_rows: list[dict[str, Any]], source_name: str) -> None:
    lines = [f"Empty-skill analysis for {source_name}", ""]
    total = sum(row["n_rows"] for row in dataset_rows)
    empties = sum(row["empty_rows"] for row in dataset_rows)
    lines.append(f"Overall rows: {total}")
    lines.append(f"Empty rows: {empties} ({safe_rate(empties, total):.4f})")
    lines.append("")
    lines.append("Per-dataset summary:")
    for row in dataset_rows:
        lines.append(
            f"- {row['dataset_display']}: empty {row['empty_rows']}/{row['n_rows']} ({row['empty_rate']:.4f}) | "
            f"empty positive rate {row['empty_positive_rate']:.4f} | nonempty positive rate {row['nonempty_positive_rate']:.4f}"
        )
        if "empty_gold_top1" in row:
            lines.append(
                f"  routing: empty top1 {row['empty_gold_top1']:.4f}, empty top4 {row['empty_gold_top4']:.4f}, "
                f"nonempty top1 {row['nonempty_gold_top1']:.4f}, nonempty top4 {row['nonempty_gold_top4']:.4f}"
            )
            lines.append(
                f"  permissive acc: empty {row['empty_permissive_acc']:.4f}, nonempty {row['nonempty_permissive_acc']:.4f}"
            )
    lines.append("")
    lines.append("Group-level text-feature summary:")
    for row in group_rows:
        lines.append(
            f"- {row['group']}: n={row['n_rows']} | positive_rate={row['positive_rate']:.4f} | "
            f"very_short_rate={row['very_short_rate']:.4f} | mention_rate={row['has_mention_rate']:.4f} | "
            f"hashtag_rate={row['has_hashtag_rate']:.4f} | url_rate={row['has_url_rate']:.4f} | "
            f"non_ascii_rate={row['has_non_ascii_rate']:.4f} | profanity_rate={row['has_profanity_rate']:.4f} | "
            f"quote_rate={row['has_quote_rate']:.4f} | avg_words={row['avg_words']:.2f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=EXPORT_DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_empty_rate(dataset_rows: list[dict[str, Any]], out_dir: Path) -> None:
    labels = [wrap_label(row["dataset_display"], width=12) for row in dataset_rows]
    values = [row["empty_rate"] for row in dataset_rows]
    fig, ax = plt.subplots(figsize=(3.15, 2.6))
    bars = ax.bar(labels, values, color="#4C78A8", alpha=0.9, edgecolor="white", linewidth=BAR_EDGE_WIDTH)
    ax.set_ylabel("Empty-skill rate")
    ax.set_ylim(0, max(values) * 1.2 if values else 1)
    ax.set_title("Empty-Skill Rate by Dataset")
    ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=GRID_WIDTH)
    ax.tick_params(axis="x", rotation=40)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
        )
    _save_figure(fig, out_dir, "empty_skill_rate_by_dataset")


def plot_positive_rate(dataset_rows: list[dict[str, Any]], out_dir: Path) -> None:
    labels = [wrap_label(row["dataset_display"], width=12) for row in dataset_rows]
    empty_vals = [row["empty_positive_rate"] for row in dataset_rows]
    nonempty_vals = [row["nonempty_positive_rate"] for row in dataset_rows]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(3.15, 2.7))
    b1 = ax.bar(
        x - width / 2,
        empty_vals,
        width=width,
        color="#E45756",
        alpha=0.9,
        label="Empty",
        edgecolor="white",
        linewidth=BAR_EDGE_WIDTH,
    )
    b2 = ax.bar(
        x + width / 2,
        nonempty_vals,
        width=width,
        color="#72B7B2",
        alpha=0.9,
        label="Nonempty",
        edgecolor="white",
        linewidth=BAR_EDGE_WIDTH,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40)
    ax.set_ylabel("Positive-label rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("Positive-Label Rate by Dataset")
    ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=GRID_WIDTH)
    ax.legend(frameon=False)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=ANNOTATION_SIZE,
            )
    _save_figure(fig, out_dir, "empty_skill_positive_rate_by_dataset")


def plot_feature_rates(group_rows: list[dict[str, Any]], out_dir: Path) -> None:
    feature_labels = [
        ("positive_rate", "positive"),
        ("very_short_rate", "very short"),
        ("has_mention_rate", "@mention"),
        ("has_hashtag_rate", "#hashtag"),
        ("has_url_rate", "URL"),
        ("has_non_ascii_rate", "non-ASCII"),
        ("has_profanity_rate", "profanity"),
        ("has_quote_rate", "quote"),
    ]
    groups = {row["group"]: row for row in group_rows}
    labels = [label for _, label in feature_labels]
    empty_vals = [groups["empty"][key] for key, _ in feature_labels]
    nonempty_vals = [groups["nonempty"][key] for key, _ in feature_labels]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(3.25, 2.85))
    ax.bar(x - width / 2, empty_vals, width=width, color="#E45756", alpha=0.9, label="Empty", edgecolor="white", linewidth=BAR_EDGE_WIDTH)
    ax.bar(x + width / 2, nonempty_vals, width=width, color="#4C78A8", alpha=0.9, label="Nonempty", edgecolor="white", linewidth=BAR_EDGE_WIDTH)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("Text-Feature Rates for Empty vs Nonempty Rows")
    ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=GRID_WIDTH)
    ax.legend(frameon=False)
    _save_figure(fig, out_dir, "empty_skill_feature_rates")


def plot_routing_metric(dataset_rows: list[dict[str, Any]], out_dir: Path, metric_key: str, title: str, stem: str) -> None:
    labels = [wrap_label(row["dataset_display"], width=12) for row in dataset_rows]
    empty_vals = [row.get(f"empty_{metric_key}", 0.0) for row in dataset_rows]
    nonempty_vals = [row.get(f"nonempty_{metric_key}", 0.0) for row in dataset_rows]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(3.15, 2.7))
    ax.bar(x - width / 2, empty_vals, width=width, color="#E45756", alpha=0.9, label="Empty", edgecolor="white", linewidth=BAR_EDGE_WIDTH)
    ax.bar(x + width / 2, nonempty_vals, width=width, color="#72B7B2", alpha=0.9, label="Nonempty", edgecolor="white", linewidth=BAR_EDGE_WIDTH)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=GRID_WIDTH)
    ax.legend(frameon=False)
    _save_figure(fig, out_dir, stem)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.skills_file)
    outputs_by_id = None
    if args.outputs_file is not None and args.outputs_file.exists():
        outputs_by_id = {row.get("id"): row for row in read_jsonl(args.outputs_file) if isinstance(row.get("id"), str)}

    dataset_rows = build_dataset_summary(rows, outputs_by_id)
    group_rows = build_group_feature_summary(rows)

    save_json(args.out_dir / "empty_skill_summary.json", {"datasets": dataset_rows, "groups": group_rows})
    save_csv(args.out_dir / "empty_skill_dataset_summary.csv", dataset_rows)
    save_csv(args.out_dir / "empty_skill_group_feature_summary.csv", group_rows)
    save_summary_txt(args.out_dir / "empty_skill_summary.txt", dataset_rows, group_rows, args.skills_file.name)
    save_examples(args.out_dir / "empty_skill_examples.txt", rows, args.example_count)

    plot_empty_rate(dataset_rows, args.out_dir)
    plot_positive_rate(dataset_rows, args.out_dir)
    plot_feature_rates(group_rows, args.out_dir)

    if outputs_by_id is not None and any("empty_gold_top1" in row for row in dataset_rows):
        plot_routing_metric(
            dataset_rows,
            args.out_dir,
            metric_key="gold_top1",
            title="Gold-Expert Top-1 Recall for Empty vs Nonempty Rows",
            stem="empty_skill_gold_top1_by_dataset",
        )
        plot_routing_metric(
            dataset_rows,
            args.out_dir,
            metric_key="gold_top4",
            title="Gold-Expert Top-4 Recall for Empty vs Nonempty Rows",
            stem="empty_skill_gold_top4_by_dataset",
        )
        plot_routing_metric(
            dataset_rows,
            args.out_dir,
            metric_key="permissive_acc",
            title="Permissive Accuracy for Empty vs Nonempty Rows",
            stem="empty_skill_permissive_accuracy_by_dataset",
        )

    print(f"Saved empty-skill analysis to: {args.out_dir}")


if __name__ == "__main__":
    main()
