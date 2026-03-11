#!/usr/bin/env python3
"""Dump routing decisions for all samples to a JSONL log (weights + selected experts)."""

from __future__ import annotations

import argparse
import importlib.util
import json
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
config_mod = _load_module(SYMBOLIC_ROOT / "config.py", "symbolic_moe_config_dump")
io_mod = _load_module(SYMBOLIC_ROOT / "io_utils.py", "symbolic_moe_io_dump")

EXPERTS = config_mod.EXPERTS
PROFILES_PATH = config_mod.PROFILES_PATH
SKILL_VOCAB = config_mod.SKILL_VOCAB
read_jsonl = io_mod.read_jsonl


def load_profiles() -> Dict:
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
    parser.add_argument(
        "--output",
        type=Path,
        default=SYMBOLIC_ROOT / "routing_log.jsonl",
        help="Where to write the routing log (JSONL).",
    )
    args = parser.parse_args()

    samples = read_jsonl(args.samples)
    profiles = load_profiles()
    strengths = compute_global_strengths(profiles)

    out_path = args.output
    with out_path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            sid = sample.get("id")
            if not isinstance(sid, str):
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

            record = {
                "id": sid,
                "dataset": sample.get("dataset"),
                "skills": skills,
                "weights": weights,
                "selected": selected,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[routing-log] Wrote {out_path}")


if __name__ == "__main__":
    main()
