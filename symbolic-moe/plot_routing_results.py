#!/usr/bin/env python3
"""Render routing-result figures from thesis summary markdown files."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


GOLD_TOPK_RE = re.compile(r"^Gold expert recall \(top-(\d)\):\s+\d+/\d+\s+\(([\d.]+)\)")
TITLE_SIZE = 10.0
LABEL_SIZE = 9.0
TICK_SIZE = 8.5
LINE_WIDTH = 1.8
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


def parse_md_table(path: Path, heading: str) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[list[str]] = []
    in_section = False
    header_seen = False
    for line in lines:
        if line.strip() == heading:
            in_section = True
            continue
        if in_section and line.startswith("## ") and line.strip() != heading:
            break
        if not in_section or not line.strip().startswith("|"):
            continue
        parts = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not header_seen:
            header_seen = True
            continue
        if parts and set("".join(parts)) == {"-"}:
            continue
        rows.append(parts)
    return rows


def parse_overall_baseline(path: Path) -> dict[str, list[float]]:
    rows = parse_md_table(path, "## Overall Routing")
    data: dict[str, list[float]] = {}
    for split, _, g1, g2, g3, g4 in rows:
        data[split] = [float(g1), float(g2), float(g3), float(g4)]
    return data


def parse_overall_nb_validation(path: Path) -> list[float]:
    rows = parse_md_table(path, "## Overall Routing")
    top1 = top2 = top3 = None
    for run, split, _, g1, g2, g3, g4 in rows:
        if split != "Validation":
            continue
        if run == "NB Top-1":
            top1 = [float(g1), float(g2), float(g3), float(g4)]
        elif run == "NB Top-2":
            top2 = [float(g1), float(g2), float(g3), float(g4)]
        elif run == "NB Top-3":
            top3 = [float(g1), float(g2), float(g3), float(g4)]
    if top1 is None or top2 is None or top3 is None:
        raise ValueError("Could not recover NB validation top-1/top-2/top-3 rows")
    return [top1[0], top2[1], top3[2], top3[3]]


def parse_gold_topks(path: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if m := GOLD_TOPK_RE.match(line):
            values[int(m.group(1))] = float(m.group(2))
    if not values:
        raise ValueError(f"No gold top-k values found in {path}")
    return values


def parse_nb_curve(top1_path: Path, top2_path: Path, top3_path: Path) -> list[float]:
    top1 = parse_gold_topks(top1_path)
    top2 = parse_gold_topks(top2_path)
    top3 = parse_gold_topks(top3_path)
    return [top1[1], top2[2], top3[3], top3[4]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render routing-result figures.")
    parser.add_argument(
        "--baseline-md",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "baseline_routing_metrics.md",
    )
    parser.add_argument(
        "--nb-md",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "nb_routing_metrics.md",
    )
    parser.add_argument(
        "--nb-val-top1",
        type=Path,
        default=Path(__file__).resolve().parent / "validation_metrics_nb_top1_from_top2.txt",
    )
    parser.add_argument(
        "--nb-val-top2",
        type=Path,
        default=Path(__file__).resolve().parent / "validation_metrics_nb_top2.txt",
    )
    parser.add_argument(
        "--nb-val-top3",
        type=Path,
        default=Path(__file__).resolve().parent / "validation_metrics_nb_top3.txt",
    )
    parser.add_argument(
        "--nb-test-top1",
        type=Path,
        default=Path(__file__).resolve().parent / "test_metrics_nb_top1_from_top2.txt",
    )
    parser.add_argument(
        "--nb-test-top2",
        type=Path,
        default=Path(__file__).resolve().parent / "test_metrics_nb_top2.txt",
    )
    parser.add_argument(
        "--nb-test-top3",
        type=Path,
        default=Path(__file__).resolve().parent / "test_metrics_nb_top3.txt",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "charts" / "results",
    )
    args = parser.parse_args()

    baseline = parse_overall_baseline(args.baseline_md)
    nb_validation = parse_nb_curve(args.nb_val_top1, args.nb_val_top2, args.nb_val_top3)
    args.outdir.mkdir(parents=True, exist_ok=True)

    ks = [1, 2, 3, 4]

    fig, ax = plt.subplots(figsize=(3.15, 2.45))
    ax.plot(ks, baseline["Validation"], marker="o", linewidth=LINE_WIDTH, label="Baseline Validation", color="#1b6ca8")
    ax.plot(ks, nb_validation, marker="D", linewidth=LINE_WIDTH, label="NB Validation", color="#c44e1a")

    ax.set_xticks(ks)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("k")
    ax.set_ylabel("Gold-expert recall")
    ax.set_title("Gold-Expert Recall at Top-k")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=False)
    pdf_path = args.outdir / "gold_expert_recall_topk.pdf"
    png_path = args.outdir / "gold_expert_recall_topk.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=EXPORT_DPI, bbox_inches="tight", pad_inches=0.02)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
