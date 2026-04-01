#!/usr/bin/env python3
"""Create per-dataset skill distribution charts from a skills JSONL pool."""
from __future__ import annotations

import argparse
import json
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASET_DISPLAY = {
    "dynahate": "DynaHate",
    "tweeteval_offensive": "TweetEval Offensive",
    "kaggle_cyberbullying": "SOSNet Cyberbullying",
    "jigsaw_threat": "Jigsaw Threat",
}

TITLE_SIZE = 10.0
LABEL_SIZE = 9.0
TICK_SIZE = 8.5
ANNOTATION_SIZE = 8.0
GRID_WIDTH = 0.8
BAR_EDGE_WIDTH = 0.6
EXPORT_DPI = 300
PANEL_FIG_WIDTH = 4.9
HEATMAP_FIG_WIDTH = 7.3

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skills-file",
        type=Path,
        default=Path(__file__).resolve().parent / "profile_pool_skills.jsonl",
        help="JSONL file containing dataset rows with predicted_skills.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "charts" / "datasets",
        help="Directory to save charts.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Top-K skills to show per dataset.",
    )
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _pretty_skill(skill: str) -> str:
    return skill.replace("_", " ")


def _wrap_label(text: str, width: int = 12) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False)) or text


def _plot_dataset(dataset: str, rows: list[dict], out_dir: Path, top_k: int) -> None:
    counts = Counter()
    empty_rows = 0

    for row in rows:
        skills = row.get("predicted_skills", [])
        if not isinstance(skills, list) or not skills:
            empty_rows += 1
            continue
        for skill in skills:
            if isinstance(skill, str) and skill.strip():
                counts[skill.strip().lower()] += 1

    if not counts:
        return

    top = counts.most_common(top_k)
    labels = [_wrap_label(_pretty_skill(skill), width=16) for skill, _ in top]
    values = [count for _, count in top]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(PANEL_FIG_WIDTH, 3.5))
    bars = ax.barh(y, values, color="#4C78A8", alpha=0.92, edgecolor="white", linewidth=BAR_EDGE_WIDTH)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Count")
    ax.grid(axis="x", linestyle="--", alpha=0.3, linewidth=GRID_WIDTH)

    display_name = DATASET_DISPLAY.get(dataset, dataset)
    ax.set_title(f"{display_name}: Inferred Skills")

    max_value = max(values) if values else 1
    ax.set_xlim(0, max_value * 1.18)
    ax.invert_yaxis()

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max_value * 0.02,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value}",
            va="center",
            ha="left",
            fontsize=ANNOTATION_SIZE,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    stem = f"{dataset}_skill_distribution"
    fig.savefig(out_dir / f"{stem}.png", dpi=EXPORT_DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _plot_heatmap(rows: list[dict], out_dir: Path, top_k: int) -> None:
    per_dataset: dict[str, Counter] = defaultdict(Counter)
    global_counts = Counter()

    for row in rows:
        dataset = row.get("dataset")
        skills = row.get("predicted_skills", [])
        if not isinstance(dataset, str) or not isinstance(skills, list):
            continue
        for skill in skills:
            if isinstance(skill, str) and skill.strip():
                skill = skill.strip().lower()
                per_dataset[dataset][skill] += 1
                global_counts[skill] += 1

    datasets = [d for d in DATASET_DISPLAY if d in per_dataset]
    skills = [skill for skill, _ in global_counts.most_common(max(top_k, 1))]
    if not datasets or not skills:
        return

    matrix = np.zeros((len(datasets), len(skills)), dtype=float)
    for i, dataset in enumerate(datasets):
        for j, skill in enumerate(skills):
            matrix[i, j] = per_dataset[dataset].get(skill, 0)

    fig_w = max(HEATMAP_FIG_WIDTH, 0.48 * len(skills) + 2.0)
    fig_h = max(3.1, 0.55 * len(datasets) + 1.15)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Count")

    ax.set_xticks(np.arange(len(skills)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels([_wrap_label(_pretty_skill(skill), width=16) for skill in skills], rotation=36, ha="right")
    ax.set_yticklabels([_wrap_label(DATASET_DISPLAY.get(dataset, dataset), width=12) for dataset in datasets])
    ax.set_title("Inferred Skill Counts by Dataset")
    ax.tick_params(axis="x", pad=3)
    ax.tick_params(axis="y", pad=3)

    fig.savefig(out_dir / "datasets_skill_heatmap.png", dpi=EXPORT_DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / "datasets_skill_heatmap.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(args.skills_file)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        dataset = row.get("dataset")
        if isinstance(dataset, str):
            grouped[dataset].append(row)

    for dataset, dataset_rows in sorted(grouped.items()):
        _plot_dataset(dataset, dataset_rows, args.out_dir, args.top_k)

    _plot_heatmap(rows, args.out_dir, args.top_k)

    print(f"Saved charts in: {args.out_dir}")
    print("Datasets:", ", ".join(sorted(grouped.keys())))


if __name__ == "__main__":
    main()
