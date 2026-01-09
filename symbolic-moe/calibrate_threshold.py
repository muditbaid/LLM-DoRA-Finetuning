#!/usr/bin/env python3
"""Calibrate routing threshold on the validation pool using per-expert predictions."""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

SYMBOLIC_ROOT = Path(__file__).resolve().parent


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


config_mod = _load_module(SYMBOLIC_ROOT / "config.py", "symbolic_moe_config")
io_mod = _load_module(SYMBOLIC_ROOT / "io_utils.py", "symbolic_moe_io")

EXPERTS = config_mod.EXPERTS
PROFILES_PATH = config_mod.PROFILES_PATH
SKILL_VOCAB = config_mod.SKILL_VOCAB
read_jsonl = io_mod.read_jsonl

# Default thresholds to sweep; override with --thresholds if desired.
DEFAULT_THRESHOLDS = [10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200]


def load_profiles() -> Dict:
    import json

    with PROFILES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_global_strengths(profiles: Dict) -> Dict[str, float]:
    strengths: Dict[str, float] = {}
    for name, profile in profiles.items():
        skill_scores = profile.get("skill_scores", {})
        strengths[name] = sum(v for v in skill_scores.values() if v > 0)
    return strengths


def _logit_from_counts(correct: int, total: int, prior: float = 1.0) -> float:
    denom = total + 2 * prior
    if denom <= 0:
        return 0.0
    p = (correct + prior) / denom
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def build_skill_odds(profiles: Dict) -> tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Return (skill_odds, priors) per expert using log-odds with Laplace smoothing."""
    skill_odds: Dict[str, Dict[str, float]] = {}
    priors: Dict[str, float] = {}
    for name, profile in profiles.items():
        stats = profile.get("stats", {})
        expert_odds: Dict[str, float] = {}
        for skill, stat in stats.items():
            correct = int(stat.get("correct", 0))
            total = int(stat.get("total", 0))
            expert_odds[skill] = _logit_from_counts(correct, total)
        skill_odds[name] = expert_odds
        total_seen = int(profile.get("total_seen", 0))
        total_correct = int(profile.get("total_correct", 0))
        priors[name] = _logit_from_counts(total_correct, total_seen)
    return skill_odds, priors


def normalize_skills(skills: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for skill in skills or []:
        if not isinstance(skill, str):
            continue
        norm = skill.strip().lower()
        if norm in SKILL_VOCAB:
            cleaned.append(norm)
    return cleaned


def collect_pairs(pred_dir: Path) -> Tuple[List[Tuple[float, bool]], int]:
    """Return (weight, is_correct) pairs across all experts and post count."""
    profiles = load_profiles()
    skill_odds, priors = build_skill_odds(profiles)

    all_pairs: List[Tuple[float, bool]] = []
    post_ids = set()

    for cfg in EXPERTS:
        pred_path = pred_dir / f"{cfg.name}.jsonl"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions file for {cfg.name}: {pred_path}")

        expert_odds = skill_odds.get(cfg.name, {})
        prior = priors.get(cfg.name, 0.0)

        for row in read_jsonl(pred_path):
            post_ids.add(row.get("id"))
            skills = normalize_skills(row.get("skill_tag") or row.get("predicted_skills") or [])
            local = sum(expert_odds.get(s, 0.0) for s in skills)
            weight = prior + local
            if weight <= 0:
                continue
            is_good = bool(row.get("is_correct"))
            all_pairs.append((weight, is_good))

    return all_pairs, len(post_ids)


def sweep_thresholds(pairs: List[Tuple[float, bool]], num_posts: int, thresholds: List[float]) -> List[Dict]:
    results: List[Dict] = []
    for T in thresholds:
        tp = fp = fn = 0
        selected_pairs = 0
        for weight, is_good in pairs:
            selected = weight >= T
            selected_pairs += int(selected)
            if selected and is_good:
                tp += 1
            elif selected and not is_good:
                fp += 1
            elif not selected and is_good:
                fn += 1
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        avg_experts = selected_pairs / num_posts if num_posts else 0.0
        results.append(
            {
                "threshold": T,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "avg_experts_per_post": avg_experts,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "predictions",
        help="Directory containing per-expert prediction JSONLs (one per expert).",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="List of thresholds to evaluate (default: %(default)s).",
    )
    args = parser.parse_args()

    pairs, num_posts = collect_pairs(args.pred_dir)
    print(f"[calibrate] Loaded {len(pairs)} (weight, is_correct) pairs across {num_posts} posts.")

    results = sweep_thresholds(pairs, num_posts, args.thresholds)
    results.sort(key=lambda r: r["threshold"])

    print("\nT\tPrec\tRec\tF1\tAvgExp/Post")
    for r in results:
        print(
            f"{r['threshold']:>4.1f}\t{r['precision']:.3f}\t{r['recall']:.3f}\t"
            f"{r['f1']:.3f}\t{r['avg_experts_per_post']:.3f}"
        )

    best = max(results, key=lambda r: r["f1"]) if results else None
    if best:
        print(
            f"\nBest by F1: T={best['threshold']} | "
            f"prec={best['precision']:.3f} rec={best['recall']:.3f} "
            f"f1={best['f1']:.3f} avg_exp/post={best['avg_experts_per_post']:.3f}"
        )


if __name__ == "__main__":
    main()
