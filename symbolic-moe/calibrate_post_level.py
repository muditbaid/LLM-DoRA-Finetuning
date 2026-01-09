#!/usr/bin/env python3
"""Post-level routing calibration: sweep relative thresholds (alpha) and report accuracy + avg experts/post."""

from __future__ import annotations

import argparse
import importlib.util
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


SYMBOLIC_ROOT = Path(__file__).resolve().parent
config_mod = _load_module(SYMBOLIC_ROOT / "config.py", "symbolic_moe_config_pl")
io_mod = _load_module(SYMBOLIC_ROOT / "io_utils.py", "symbolic_moe_io_pl")
EXPERTS = config_mod.EXPERTS
PROFILES_PATH = config_mod.PROFILES_PATH
SKILL_VOCAB = config_mod.SKILL_VOCAB
read_jsonl = io_mod.read_jsonl

DEFAULT_ALPHAS = [0.4, 0.5, 0.6, 0.7, 0.8]


def normalize_label(text: str, mode: str) -> str:
    t = " ".join(text.strip().lower().split())
    if mode == "bully":
        if "not_bully" in t or "not bully" in t:
            return "not_bully"
        if "label: bully" in t or "bully" in t:
            return "bully"
    return t


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


def load_predictions(pred_dir: Path) -> Dict[str, Dict[str, str]]:
    """Return mapping expert -> {sample_id -> normalized_prediction}."""
    per_expert: Dict[str, Dict[str, str]] = {}
    for cfg in EXPERTS:
        path = pred_dir / f"{cfg.name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing predictions file for expert {cfg.name}: {path}")
        mapping: Dict[str, str] = {}
        for row in read_jsonl(path):
            pid = row.get("id")
            pred = row.get("normalized_prediction")
            if isinstance(pid, str) and isinstance(pred, str):
                mapping[pid] = pred.strip().lower()
        per_expert[cfg.name] = mapping
    return per_expert


def _dataset_normalizer(dataset: str) -> str:
    for cfg in EXPERTS:
        if cfg.dataset == dataset:
            return cfg.normalizer
    return "simple"


def _base_label_if_negative(label: str) -> str | None:
    txt = label.strip().lower()
    if txt.startswith("not "):
        return txt[4:].strip()
    if txt.startswith("not_"):
        return txt[4:].lstrip("_")
    return None


def record_correct(gold_label: str, predictions: List[dict], dataset: str) -> bool:
    """Match logic copied from evaluate_outputs.record_correct to keep metrics aligned."""
    normalizer = _dataset_normalizer(dataset)
    gold = normalize_label(gold_label or "", normalizer)
    preds: List[str] = []
    for entry in predictions:
        if not isinstance(entry, dict):
            continue
        norm = entry.get("normalized_prediction")
        raw = entry.get("raw_prediction")
        if isinstance(norm, str) and norm.strip():
            preds.append(norm.strip().lower())
        elif isinstance(raw, str) and raw.strip():
            preds.append(raw.strip().lower())

    if gold in preds:
        return True

    base = _base_label_if_negative(gold)
    if base:
        return base not in preds

    return False


def build_weights(
    samples: List[dict],
    profiles: Dict,
    skill_odds: Dict[str, Dict[str, float]],
    priors: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """Return weights[expert][sample_id] = weight."""
    weights: Dict[str, Dict[str, float]] = defaultdict(dict)
    skill_cache: Dict[str, List[str]] = {}

    for sample in samples:
        sid = sample.get("id")
        if not isinstance(sid, str):
            continue
        skills = normalize_skills(sample.get(config_mod.SKILL_FIELD) or [])
        skill_cache[sid] = skills

    for cfg in EXPERTS:
        expert_odds = skill_odds.get(cfg.name, {})
        prior = priors.get(cfg.name, 0.0)
        for sid, skills in skill_cache.items():
            if not skills:
                continue
            local = sum(expert_odds.get(s, 0.0) for s in skills)
            weight = prior + local
            if weight > 0:
                weights[cfg.name][sid] = weight

    return weights


def select_experts(weights_for_sample: Dict[str, float], alpha: float) -> List[str]:
    if not weights_for_sample:
        return []
    max_w = max(weights_for_sample.values())
    selected = [e for e, w in weights_for_sample.items() if w >= alpha * max_w and w > 0]
    if not selected:
        # fallback: force the best expert if all fell below alpha
        best = max(weights_for_sample.items(), key=lambda kv: kv[1])[0]
        selected = [best]
    return selected


def sweep_alphas(
    samples: List[dict],
    weights: Dict[str, Dict[str, float]],
    preds: Dict[str, Dict[str, str]],
    alphas: List[float],
) -> List[Dict]:
    results: List[Dict] = []
    total_posts = len(samples)
    for alpha in alphas:
        correct = 0
        selected_total = 0
        for sample in samples:
            sid = sample.get("id")
            if not isinstance(sid, str):
                continue
            # gather candidate weights for this sample
            weights_for_sample = {e: wmap[sid] for e, wmap in weights.items() if sid in wmap}
            selected = select_experts(weights_for_sample, alpha)
            selected_total += len(selected)

            selected_preds = []
            for expert in selected:
                pred = preds.get(expert, {}).get(sid)
                if pred:
                    selected_preds.append({"normalized_prediction": pred})
            ok = record_correct(sample.get("output", ""), selected_preds, sample.get("dataset", ""))
            correct += int(ok)

        acc = correct / total_posts if total_posts else 0.0
        avg_experts = selected_total / total_posts if total_posts else 0.0
        results.append({"alpha": alpha, "accuracy": acc, "avg_experts_per_post": avg_experts})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=Path,
        default=SYMBOLIC_ROOT / "validation_pool_skills.jsonl",
        help="Validation pool with predicted_skills.",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=SYMBOLIC_ROOT / "predictions",
        help="Directory of per-expert prediction JSONLs.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
        help="Relative thresholds to sweep (fraction of max weight per post).",
    )
    args = parser.parse_args()

    samples = read_jsonl(args.samples)
    preds = load_predictions(args.pred_dir)
    profiles = load_profiles()
    skill_odds, priors = build_skill_odds(profiles)
    weights = build_weights(samples, profiles, skill_odds, priors)

    results = sweep_alphas(samples, weights, preds, args.alphas)
    results.sort(key=lambda r: r["alpha"])

    print("alpha\tacc\tavg_experts/post")
    for r in results:
        print(f"{r['alpha']:.2f}\t{r['accuracy']:.3f}\t{r['avg_experts_per_post']:.3f}")

    best = max(results, key=lambda r: r["accuracy"]) if results else None
    if best:
        print(
            f"\nBest by accuracy: alpha={best['alpha']:.2f} | "
            f"acc={best['accuracy']:.3f} | avg_exp/post={best['avg_experts_per_post']:.3f}"
        )


if __name__ == "__main__":
    main()
