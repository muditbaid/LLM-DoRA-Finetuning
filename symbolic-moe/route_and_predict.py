#!/usr/bin/env python3
"""
Route samples to experts using learned profiles and return each expert's decision.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

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


build_profiles_mod = _load_module(SYMBOLIC_ROOT / "build_profiles.py", "symbolic_moe_build_profiles_route")
config_mod = _load_module(SYMBOLIC_ROOT / "config.py", "symbolic_moe_config_route")
io_mod = _load_module(SYMBOLIC_ROOT / "io_utils.py", "symbolic_moe_io_route")
model_mod = _load_module(SYMBOLIC_ROOT / "model_utils.py", "symbolic_moe_model_route")

normalize_label = build_profiles_mod.normalize_label
EXPERTS = config_mod.EXPERTS
PROFILES_PATH = config_mod.PROFILES_PATH
SKILL_FIELD = config_mod.SKILL_FIELD
SKILL_VOCAB = config_mod.SKILL_VOCAB
TEST_SAMPLE = config_mod.TEST_SAMPLE
read_jsonl = io_mod.read_jsonl
write_jsonl = io_mod.write_jsonl
ExpertModel = model_mod.ExpertModel
SharedModelRuntime = model_mod.SharedModelRuntime

ALPHA = 0.4  # relative threshold fraction of max weight



def load_profiles():
    with PROFILES_PATH.open("r", encoding="utf-8") as f:
        profiles = json.load(f)
    print(f"[routing] Loaded profiles from {PROFILES_PATH}")
    return profiles


def _logit_from_counts(correct: int, total: int, prior: float = 1.0) -> float:
    denom = total + 2 * prior
    if denom <= 0:
        return 0.0
    p = (correct + prior) / denom
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def build_skill_odds(profiles: dict) -> tuple[dict, dict]:
    """Return (skill_odds, priors) per expert using log-odds with Laplace smoothing."""
    skill_odds = {}
    priors = {}
    for name, profile in profiles.items():
        stats = profile.get("stats", {})
        expert_odds = {}
        for skill, stat in stats.items():
            correct = int(stat.get("correct", 0))
            total = int(stat.get("total", 0))
            expert_odds[skill] = _logit_from_counts(correct, total)
        skill_odds[name] = expert_odds
        total_seen = int(profile.get("total_seen", 0))
        total_correct = int(profile.get("total_correct", 0))
        priors[name] = _logit_from_counts(total_correct, total_seen)
    return skill_odds, priors


def main():
    parser = argparse.ArgumentParser(description="Route samples via Symbolic-MoE profiles.")
    parser.add_argument(
        "--input",
        type=Path,
        default=TEST_SAMPLE,
        help="JSONL file containing samples (optionally with predicted_skills).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROFILES_PATH.parent / "test_sample_outputs.jsonl",
        help="Where to save routed predictions.",
    )
    args = parser.parse_args()

    samples = read_jsonl(args.input)
    profiles = load_profiles()

    skill_odds, priors = build_skill_odds(profiles)

    default_map = {cfg.label: cfg.name for cfg in EXPERTS}

    assignments = defaultdict(list)  # expert_name -> list of (sample_idx, weight)
    routed_info = []
    results = [dict(sample) for sample in samples]

    for idx, sample in enumerate(samples):
        raw_skills = sample.get(SKILL_FIELD)
        if raw_skills is None:
            existing = sample.get("skill_tag")
            if isinstance(existing, list):
                raw_skills = existing
            elif isinstance(existing, str):
                raw_skills = [] if existing in ("none", "", None) else [existing]
            else:
                raw_skills = []
        mapped_skills = []
        for skill in raw_skills:
            norm = skill.lower().strip()
            if norm in SKILL_VOCAB:
                mapped_skills.append(norm)
        print(f"[routing] Sample {sample.get('id')} raw_skills={raw_skills} mapped={mapped_skills}")
        candidates = []
        for cfg in EXPERTS:
            profile = profiles.get(cfg.name)
            if not profile:
                continue

            expert_odds = skill_odds.get(cfg.name, {})
            prior = priors.get(cfg.name, 0.0)
            if mapped_skills:
                local_score = sum(expert_odds.get(skill, 0.0) for skill in mapped_skills)
                weight = prior + local_score
            else:
                # No skills inferred: fall back to expert prior only
                weight = prior

            if weight > 0:
                candidates.append((cfg.name, weight))
            print(f"[routing] weight for {cfg.name}: {weight}")

        candidates.sort(key=lambda x: x[1], reverse=True)

        selected = []
        if mapped_skills:
            if candidates:
                max_w = candidates[0][1]
                selected = [(n, w) for n, w in candidates if w >= ALPHA * max_w and w > 0]
                if not selected and candidates:
                    selected = [candidates[0]]
        else:
            # no skills -> send to all candidates
            selected = candidates
        print(f"[routing] Selected {len(selected)} experts for {sample.get('id')}: {selected}")


        routed_info.append(
            {
                "id": sample.get("id"),
                "raw_skills": raw_skills,
                "mapped_skills": mapped_skills,
                "assigned_experts": [name for name, _ in selected],
            }
        )
        for name, weight in selected:
            assignments[name].append((idx, weight))

    results = [dict(sample) for sample in samples]
    # Route-and-predict runs in 16-bit mode (no 8-bit quantization).
    runtime = SharedModelRuntime(use_quantization=False)
    try:
        for cfg in EXPERTS:
            assigned = assignments.get(cfg.name, [])
            if not assigned:
                continue
            print(f"[symbolic-moe] Running expert {cfg.name} on {len(assigned)} routed samples…")
            model = ExpertModel(cfg, runtime=runtime)
            for sample_idx, weight in assigned:
                rec = samples[sample_idx]
                prompt = model.build_prompt(rec.get("system", ""), rec.get("instruction", ""), rec.get("input", ""))
                pred_text, confidence = model.predict_with_confidence(prompt)
                norm_pred = normalize_label(pred_text, cfg.normalizer)
                record = results[sample_idx]
                record.setdefault("predictions", []).append(
                    {
                        "expert": cfg.name,
                        "weight": weight,
                        "label_confidence": confidence,
                        "raw_prediction": pred_text.strip(),
                        "normalized_prediction": norm_pred,
                    }
                )
            model.close()
            torch.cuda.empty_cache()
    finally:
        runtime.close()

    write_jsonl(args.output, results)
    print(f"[symbolic-moe] Routed outputs saved to {args.output}")


if __name__ == "__main__":
    main()
