#!/usr/bin/env python3
"""Plot skill-distribution charts for each routed expert."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _collect_stats(path: Path):
    skill_counts_by_expert: dict[str, Counter[str]] = defaultdict(Counter)
    routed_samples_by_expert: Counter[str] = Counter()
    all_skills: set[str] = set()

    for row in _read_jsonl(path):
        skills = row.get("predicted_skills") or []
        if not isinstance(skills, list):
            skills = []
        skills = sorted({str(s).strip() for s in skills if str(s).strip()})
        if not skills:
            skills = ["<none>"]

        predictions = row.get("predictions") or []
        if not isinstance(predictions, list):
            continue

        experts = []
        for pred in predictions:
            if not isinstance(pred, dict):
                continue
            expert = pred.get("expert")
            if isinstance(expert, str) and expert.strip():
                experts.append(expert.strip())
        experts = sorted(set(experts))

        for expert in experts:
            routed_samples_by_expert[expert] += 1
            skill_counts_by_expert[expert].update(skills)
            all_skills.update(skills)

    return skill_counts_by_expert, routed_samples_by_expert, sorted(all_skills)


def _plot_per_expert(
    out_dir: Path,
    skill_counts_by_expert: dict[str, Counter[str]],
    routed_samples_by_expert: Counter[str],
    top_k: int,
) -> None:
    for expert in sorted(skill_counts_by_expert.keys()):
        counts = skill_counts_by_expert[expert]
        total = routed_samples_by_expert[expert]
        if total == 0:
            continue

        top_items = counts.most_common(top_k)
        labels = [k for k, _ in top_items]
        values = [100.0 * v / total for _, v in top_items]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(labels, values, color="#4C78A8")
        ax.set_title(f"Skill Distribution for Expert: {expert} (n={total})")
        ax.set_ylabel("% of routed samples containing skill")
        ax.set_ylim(0, max(values) * 1.15 if values else 1)
        ax.tick_params(axis="x", rotation=40)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        for bar, pct in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        fig.tight_layout()
        out_path = out_dir / f"{expert}_skill_distribution.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)


def _plot_heatmap(
    out_dir: Path,
    skill_counts_by_expert: dict[str, Counter[str]],
    routed_samples_by_expert: Counter[str],
    all_skills: list[str],
    top_k_skills: int,
) -> None:
    experts = sorted(skill_counts_by_expert.keys())
    if not experts:
        return

    global_counts = Counter()
    for counter in skill_counts_by_expert.values():
        global_counts.update(counter)
    skills = [s for s, _ in global_counts.most_common(top_k_skills)]

    matrix = np.zeros((len(experts), len(skills)), dtype=float)
    for i, expert in enumerate(experts):
        total = max(1, routed_samples_by_expert[expert])
        for j, skill in enumerate(skills):
            matrix[i, j] = 100.0 * skill_counts_by_expert[expert][skill] / total

    fig_w = max(12, 0.8 * len(skills))
    fig_h = max(4, 0.8 * len(experts))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("% of routed samples containing skill")

    ax.set_xticks(np.arange(len(skills)))
    ax.set_yticks(np.arange(len(experts)))
    ax.set_xticklabels(skills, rotation=45, ha="right")
    ax.set_yticklabels(experts)
    ax.set_title("Skill Presence by Expert (Top Skills)")

    fig.tight_layout()
    fig.savefig(out_dir / "experts_skill_heatmap.png", dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "hatebench_pool_outputs.jsonl",
        help="JSONL file with `predicted_skills` and per-expert `predictions`.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "charts",
        help="Directory to write chart images.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Top-k skills shown per expert chart.",
    )
    parser.add_argument(
        "--heatmap-top-k",
        type=int,
        default=16,
        help="Top-k global skills shown in the heatmap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    skill_counts_by_expert, routed_samples_by_expert, all_skills = _collect_stats(args.input)
    _plot_per_expert(args.out_dir, skill_counts_by_expert, routed_samples_by_expert, args.top_k)
    _plot_heatmap(
        args.out_dir,
        skill_counts_by_expert,
        routed_samples_by_expert,
        all_skills,
        args.heatmap_top_k,
    )

    print(f"Wrote charts to: {args.out_dir}")
    print(f"Experts: {', '.join(sorted(skill_counts_by_expert.keys()))}")


if __name__ == "__main__":
    main()
