#!/usr/bin/env python3
"""Render a thesis-style expert profile excerpt directly from profiles.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


TITLE_SIZE = 10.0
LABEL_SIZE = 9.0
TICK_SIZE = 8.5
BODY_SIZE = 8.5
SMALL_SIZE = 8.0
EXPORT_DPI = 300
BG = "#FCFBF8"
HEADER_BG = "#E7EEF6"
HEADER_EDGE = "#9DB3C7"
CARD_BG = "#F7F4EE"
CARD_EDGE = "#C9BFB1"
TEXT = "#222222"
SUBTLE = "#5D6670"
ACCENT = "#355C7D"
NOTE_BG = "#EEF4EA"
NOTE_EDGE = "#A7BC9A"

plt.rcParams.update(
    {
        "font.size": BODY_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


DEFAULT_SKILLS = [
    "dehumanization",
    "coded_hostility",
    "stereotype_invocation",
    "identity_targeting",
    "general_insult",
    "threatening_language",
    "sexual_harassment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path(__file__).resolve().parent / "profiles.json",
        help="Path to profiles.json",
    )
    parser.add_argument(
        "--expert",
        default="dynahate_hate",
        help="Expert key to render",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "charts" / "profiles",
        help="Output directory",
    )
    return parser.parse_args()


def excerpt_lines(profile: dict, skills: list[str]) -> tuple[list[str], list[str], list[str]]:
    skill_scores = profile["skill_scores"]
    raw_margin = profile["raw_skill_margin"]
    stats = profile["stats"]

    score_lines = [f'  "{skill}": {skill_scores[skill]:.4f},' for skill in skills if skill in skill_scores]
    margin_lines = [f'  "{skill}": {raw_margin[skill]},' for skill in skills if skill in raw_margin]
    stat_lines = []
    for skill in skills:
        if skill in stats:
            stat_lines.append(
                f'  "{skill}": {{"correct": {stats[skill]["correct"]}, "total": {stats[skill]["total"]}}},'
            )
    if score_lines:
        score_lines[-1] = score_lines[-1].rstrip(",")
    if margin_lines:
        margin_lines[-1] = margin_lines[-1].rstrip(",")
    if stat_lines:
        stat_lines[-1] = stat_lines[-1].rstrip(",")
    return score_lines, margin_lines, stat_lines


def draw_block(fig, ax, x: float, y: float, title: str, lines: list[str], width: float, height: float) -> None:
    box = FancyBboxPatch(
        (x, y - height),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=0.8,
        edgecolor=CARD_EDGE,
        facecolor=CARD_BG,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    fig.text(x + 0.012, y - 0.018, title, fontsize=BODY_SIZE, fontweight="bold", color=ACCENT, ha="left", va="top")
    block_text = "{\n" + "\n".join(lines) + "\n  ...\n}"
    fig.text(
        x + 0.012,
        y - 0.062,
        block_text,
        family="monospace",
        fontsize=SMALL_SIZE,
        color=TEXT,
        ha="left",
        va="top",
    )


def draw_metric_chip(fig, ax, x: float, y: float, label: str, value: str, width: float) -> None:
    chip = FancyBboxPatch(
        (x, y - 0.07),
        width,
        0.07,
        boxstyle="round,pad=0.01,rounding_size=0.01",
        linewidth=0.7,
        edgecolor=HEADER_EDGE,
        facecolor="white",
        transform=ax.transAxes,
    )
    ax.add_patch(chip)
    fig.text(x + 0.012, y - 0.018, label, fontsize=SMALL_SIZE, color=SUBTLE, ha="left", va="top")
    fig.text(x + 0.012, y - 0.042, value, fontsize=BODY_SIZE, color=TEXT, ha="left", va="top", fontweight="bold")


def main() -> None:
    args = parse_args()
    profiles = json.loads(args.profiles.read_text(encoding="utf-8"))
    if args.expert not in profiles:
        raise KeyError(f"Expert {args.expert!r} not found in {args.profiles}")

    profile = profiles[args.expert]
    stats = profile["stats"]

    excerpt_skills = [skill for skill in DEFAULT_SKILLS[:3] if skill in stats]
    score_lines, margin_lines, stat_lines = excerpt_lines(profile, excerpt_skills)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.0, 4.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.patch.set_facecolor(BG)

    header = FancyBboxPatch(
        (0.05, 0.80),
        0.90,
        0.15,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=0.9,
        edgecolor=HEADER_EDGE,
        facecolor=HEADER_BG,
        transform=ax.transAxes,
    )
    ax.add_patch(header)

    fig.text(0.07, 0.93, "Example Expert Profile Excerpt", fontsize=TITLE_SIZE, fontweight="bold", color=TEXT, ha="left", va="top")
    fig.text(0.07, 0.885, args.expert, fontsize=BODY_SIZE, fontweight="bold", color=ACCENT, ha="left", va="top")
    fig.text(0.07, 0.855, f"Dataset: {profile['dataset']}   |   Label: {profile['label']}", fontsize=BODY_SIZE, color=SUBTLE, ha="left", va="top")

    draw_metric_chip(fig, ax, 0.58, 0.93, "Total seen", f"{profile['total_seen']}", 0.11)
    draw_metric_chip(fig, ax, 0.71, 0.93, "Total correct", f"{profile['total_correct']}", 0.13)
    draw_metric_chip(fig, ax, 0.86, 0.93, "Accuracy", f"{profile['accuracy']:.4f}", 0.08)

    draw_block(fig, ax, 0.05, 0.74, "skill_scores", score_lines, 0.28, 0.38)
    draw_block(fig, ax, 0.36, 0.74, "raw_skill_margin", margin_lines, 0.28, 0.38)
    draw_block(fig, ax, 0.67, 0.74, "stats", stat_lines, 0.28, 0.38)

    note = FancyBboxPatch(
        (0.05, 0.08),
        0.90,
        0.10,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=0.8,
        edgecolor=NOTE_EDGE,
        facecolor=NOTE_BG,
        transform=ax.transAxes,
    )
    ax.add_patch(note)

    fig.text(
        0.07,
        0.155,
        "Interpretation: the profile stores a compact set of skill-conditioned competence statistics",
        fontsize=SMALL_SIZE,
        color=TEXT,
        ha="left",
        va="top",
    )
    fig.text(
        0.07,
        0.122,
        "for each expert, including skill scores, raw margins, and correctness counts.",
        fontsize=SMALL_SIZE,
        color=TEXT,
        ha="left",
        va="top",
    )

    stem = f"{args.expert}_profile_card"
    fig.savefig(args.out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.out_dir / f"{stem}.png", dpi=EXPORT_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {args.out_dir / f'{stem}.pdf'}")


if __name__ == "__main__":
    main()
