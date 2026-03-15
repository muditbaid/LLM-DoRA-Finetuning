#!/usr/bin/env python3
"""Create per-expert skill distribution charts from symbolic-moe/profiles.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path(__file__).resolve().parent / "profiles.json",
        help="Path to profiles.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "charts" / "profiles",
        help="Directory to save charts",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Top-K skills per expert chart",
    )
    return parser.parse_args()


def _load_profiles(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("profiles.json must be a JSON object keyed by expert name.")
    return data


def _plot_expert(expert: str, profile: dict, out_dir: Path, top_k: int) -> None:
    stats = profile.get("stats", {})
    total_seen = int(profile.get("total_seen", 0) or 0)
    if not isinstance(stats, dict) or total_seen <= 0:
        return

    rows = []
    for skill, s in stats.items():
        if not isinstance(s, dict):
            continue
        total = int(s.get("total", 0) or 0)
        correct = int(s.get("correct", 0) or 0)
        if total <= 0:
            continue
        rows.append((skill, total, correct))

    if not rows:
        return

    rows.sort(key=lambda x: x[1], reverse=True)
    rows = rows[:top_k]

    labels = [r[0] for r in rows]
    totals = [r[1] for r in rows]
    corrects = [r[2] for r in rows]

    fig, ax1 = plt.subplots(figsize=(13, 6))
    x = np.arange(len(labels))
    width = 0.42
    bars_total = ax1.bar(x - width / 2, totals, width=width, color="#4C78A8", alpha=0.9, label="total")
    bars_correct = ax1.bar(x + width / 2, corrects, width=width, color="#72B7B2", alpha=0.9, label="correct")
    ax1.set_ylabel("Count")
    ax1.set_ylim(0, max(totals) * 1.2 if totals else 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.tick_params(axis="x", rotation=38)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend()

    title = f"{expert} | top {len(rows)} skills | total_seen={total_seen} | total_correct={int(profile.get('total_correct', 0) or 0)}"
    ax1.set_title(title)

    for b, v in zip(bars_total, totals):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height(),
            f"{v}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for b, v in zip(bars_correct, corrects):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height(),
            f"{v}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    out_path = out_dir / f"{expert}_skill_distribution.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_heatmap(profiles: dict, out_dir: Path, top_k: int) -> None:
    experts = sorted(profiles.keys())
    if not experts:
        return

    skill_totals_global = {}
    for expert in experts:
        stats = profiles[expert].get("stats", {})
        if not isinstance(stats, dict):
            continue
        for skill, s in stats.items():
            if not isinstance(s, dict):
                continue
            skill_totals_global[skill] = skill_totals_global.get(skill, 0) + int(s.get("total", 0) or 0)

    skills = [k for k, _ in sorted(skill_totals_global.items(), key=lambda kv: kv[1], reverse=True)[: max(top_k, 1)]]
    if not skills:
        return

    matrix = np.zeros((len(experts), len(skills)), dtype=float)
    for i, expert in enumerate(experts):
        p = profiles[expert]
        stats = p.get("stats", {})
        for j, skill in enumerate(skills):
            total = int((stats.get(skill, {}) or {}).get("total", 0) or 0)
            matrix[i, j] = total

    fig_w = max(12, 0.85 * len(skills))
    fig_h = max(4, 0.9 * len(experts))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Skill count (total)")

    ax.set_xticks(np.arange(len(skills)))
    ax.set_yticks(np.arange(len(experts)))
    ax.set_xticklabels(skills, rotation=45, ha="right")
    ax.set_yticklabels(experts)
    ax.set_title("Skill Counts by Expert (from profiles.json)")

    fig.tight_layout()
    fig.savefig(out_dir / "profiles_skill_heatmap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    profiles = _load_profiles(args.profiles)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for expert, profile in sorted(profiles.items()):
        if isinstance(profile, dict):
            _plot_expert(expert, profile, args.out_dir, args.top_k)
    _plot_heatmap(profiles, args.out_dir, args.top_k)

    print(f"Saved charts in: {args.out_dir}")
    print("Experts:", ", ".join(sorted(profiles.keys())))


if __name__ == "__main__":
    main()
