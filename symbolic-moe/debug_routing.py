#!/usr/bin/env python3
"""Inspect routing decisions: show cases where the gold expert was not selected."""

from __future__ import annotations

import argparse
import importlib.util
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
config_mod = _load_module(SYMBOLIC_ROOT / "config.py", "symbolic_moe_config_dbg")
io_mod = _load_module(SYMBOLIC_ROOT / "io_utils.py", "symbolic_moe_io_dbg")

EXPERTS = config_mod.EXPERTS
PROFILES_PATH = config_mod.PROFILES_PATH
SKILL_VOCAB = config_mod.SKILL_VOCAB
read_jsonl = io_mod.read_jsonl


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


def normalize_skills(skills: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for skill in skills or []:
        if not isinstance(skill, str):
            continue
        norm = skill.strip().lower()
        if norm in SKILL_VOCAB:
            cleaned.append(norm)
    return cleaned


def select_experts_alpha(weights_for_sample: List[Tuple[str, float]], alpha: float, max_experts: int) -> List[Tuple[str, float]]:
    if not weights_for_sample:
        return []
    max_w = max(w for _, w in weights_for_sample)
    chosen = [(e, w) for e, w in weights_for_sample if w > 0 and w >= alpha * max_w]
    if not chosen:
        # fallback: force top-1 if nothing passes alpha
        top = max(weights_for_sample, key=lambda x: x[1])
        chosen = [top]
    chosen.sort(key=lambda x: x[1], reverse=True)
    return chosen[:max_experts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=Path,
        default=SYMBOLIC_ROOT / "validation_pool_skills.jsonl",
        help="JSONL with predicted_skills per sample.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Relative threshold (fraction of per-post max weight).",
    )
    parser.add_argument("--max-experts", type=int, default=3, help="Max experts to keep per sample.")
    parser.add_argument("--limit", type=int, default=20, help="How many misses to print.")
    args = parser.parse_args()

    samples = read_jsonl(args.samples)
    profiles = load_profiles()
    strengths = compute_global_strengths(profiles)

    dataset_to_expert = {cfg.dataset: cfg.name for cfg in EXPERTS}

    misses = []
    for sample in samples:
        sid = sample.get("id")
        dataset = sample.get("dataset")
        gold_expert = dataset_to_expert.get(dataset)
        if not gold_expert or not isinstance(sid, str):
            continue

        skills = normalize_skills(sample.get(config_mod.SKILL_FIELD) or [])

        weights: List[Tuple[str, float]] = []
        for cfg in EXPERTS:
            profile = profiles.get(cfg.name, {})
            skill_scores = profile.get("skill_scores", {})
            strength = strengths.get(cfg.name, 0.0)
            if strength <= 0:
                continue
            local = sum(skill_scores.get(s, 0) for s in skills)
            w = local * strength
            if w > 0:
                weights.append((cfg.name, w))

        weights.sort(key=lambda x: x[1], reverse=True)
        selected = select_experts_alpha(weights, args.alpha, args.max_experts)
        selected_names = {e for e, _ in selected}

        if gold_expert not in selected_names:
            misses.append(
                {
                    "id": sid,
                    "dataset": dataset,
                    "skills": skills,
                    "weights": weights[:5],
                    "selected": selected,
                    "gold_expert": gold_expert,
                }
            )
        if len(misses) >= args.limit:
            break

    print(f"Total samples: {len(samples)} | Misses (gold expert not selected): {len(misses)}")
    for m in misses:
        print("\n---")
        print(f"id: {m['id']} | dataset: {m['dataset']} | gold: {m['gold_expert']}")
        print(f"skills: {m['skills']}")
        print("top weights:")
        for e, w in m["weights"]:
            print(f"  {e:20s} -> {w:.2f}")
        print("selected:")
        for e, w in m["selected"]:
            print(f"  {e:20s} -> {w:.2f}")


if __name__ == "__main__":
    main()
