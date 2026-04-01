#!/usr/bin/env python3
"""Create per-expert skill distribution charts from symbolic-moe/profiles.json."""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TITLE_SIZE = 10.0
LABEL_SIZE = 9.0
TICK_SIZE = 8.5
ANNOTATION_SIZE = 8.0
GRID_WIDTH = 0.8
BAR_EDGE_WIDTH = 0.6
EXPORT_DPI = 300
PANEL_FIG_WIDTH = 3.15
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


def _wrap_label(text: str, width: int = 12) -> str:
    return "\n".join(textwrap.wrap(text.replace("_", " "), width=width, break_long_words=False)) or text


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

    labels = [_wrap_label(r[0], width=12) for r in rows]
    totals = [r[1] for r in rows]
    corrects = [r[2] for r in rows]

    fig, ax1 = plt.subplots(figsize=(PANEL_FIG_WIDTH, 2.9))
    x = np.arange(len(labels))
    width = 0.42
    bars_total = ax1.bar(x - width / 2, totals, width=width, color="#4C78A8", alpha=0.9, label="total", edgecolor="white", linewidth=BAR_EDGE_WIDTH)
    bars_correct = ax1.bar(x + width / 2, corrects, width=width, color="#72B7B2", alpha=0.9, label="correct", edgecolor="white", linewidth=BAR_EDGE_WIDTH)
    ax1.set_ylabel("Count")
    ax1.set_ylim(0, max(totals) * 1.2 if totals else 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.tick_params(axis="x", rotation=42)
    ax1.grid(axis="y", linestyle="--", alpha=0.3, linewidth=GRID_WIDTH)
    ax1.legend(frameon=False)

    ax1.set_title(f"{expert}: Skill Counts")

    for b, v in zip(bars_total, totals):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height(),
            f"{v}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
        )
    for b, v in zip(bars_correct, corrects):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height(),
            f"{v}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
        )

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    out_path = out_dir / f"{expert}_skill_distribution.png"
    fig.savefig(out_path, dpi=EXPORT_DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / f"{expert}_skill_distribution.pdf", bbox_inches="tight", pad_inches=0.02)
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

    fig_w = max(HEATMAP_FIG_WIDTH, 0.48 * len(skills) + 2.0)
    fig_h = max(3.1, 0.55 * len(experts) + 1.15)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Count")

    ax.set_xticks(np.arange(len(skills)))
    ax.set_yticks(np.arange(len(experts)))
    ax.set_xticklabels([_wrap_label(skill, width=16) for skill in skills], rotation=36, ha="right")
    ax.set_yticklabels([expert.replace("_", " ") for expert in experts])
    ax.set_title("Profile Skill Counts by Expert")
    ax.tick_params(axis="x", pad=3)
    ax.tick_params(axis="y", pad=3)

    fig.savefig(out_dir / "profiles_skill_heatmap.png", dpi=EXPORT_DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / "profiles_skill_heatmap.pdf", bbox_inches="tight", pad_inches=0.02)
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
