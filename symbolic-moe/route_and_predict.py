#!/usr/bin/env python3
"""
Route samples to experts using learned profiles and return each expert's decision.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from .build_profiles import normalize_label
from .config import EXPERTS, PROFILES_PATH, SKILL_TAGS, SKILL_TO_PROFILE, TEST_SAMPLE
from .io_utils import read_jsonl, write_jsonl
from .model_utils import ExpertModel

MAX_EXPERTS_PER_SAMPLE = 2


def load_profiles():
    with PROFILES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    default_map = {cfg.skill_tag: cfg.name for cfg in EXPERTS}

    assignments = defaultdict(list)  # expert_name -> list of (sample_idx, weight)
    routed_info = []
    results = [dict(sample) for sample in samples]

    for idx, sample in enumerate(samples):
        raw_skills = sample.get("predicted_skills")
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
            mapped = SKILL_TO_PROFILE.get(norm, norm)
            if mapped in SKILL_TAGS and mapped != "none":
                mapped_skills.append(mapped)

        if not mapped_skills:
            results[idx]["predictions"] = []
            routed_info.append(
                {
                    "id": sample.get("id"),
                    "raw_skills": raw_skills,
                    "mapped_skills": [],
                    "assigned_experts": [],
                }
            )
            continue

        candidates = []
        for cfg in EXPERTS:
            profile = profiles.get(cfg.name)
            if not profile:
                continue
            score = sum(profile["skill_scores"].get(skill, 0) for skill in mapped_skills)
            weight = score * profile.get("accuracy", 0.0)
            if weight > 0:
                candidates.append((cfg.name, weight))
        candidates.sort(key=lambda x: x[1], reverse=True)

        if not candidates:
            # fallback to default model for the first mapped skill
            fallback = default_map.get(mapped_skills[0])
            if fallback:
                candidates = [(fallback, 0.0)]

        candidates = candidates[:MAX_EXPERTS_PER_SAMPLE]
        routed_info.append(
            {
                "id": sample.get("id"),
                "raw_skills": raw_skills,
                "mapped_skills": mapped_skills,
                "assigned_experts": [name for name, _ in candidates],
            }
        )
        for name, weight in candidates:
            assignments[name].append((idx, weight))

    results = [dict(sample) for sample in samples]
    for cfg in EXPERTS:
        assigned = assignments.get(cfg.name, [])
        if not assigned:
            continue
        print(f"[symbolic-moe] Running expert {cfg.name} on {len(assigned)} routed samples…")
        model = ExpertModel(cfg)
        for sample_idx, weight in assigned:
            rec = samples[sample_idx]
            prompt = model.build_prompt(rec.get("system", ""), rec.get("instruction", ""), rec.get("input", ""))
            pred_text = model.predict(prompt)
            norm_pred = normalize_label(pred_text, cfg.normalizer)
            record = results[sample_idx]
            record.setdefault("predictions", []).append(
                {
                    "expert": cfg.name,
                    "weight": weight,
                    "raw_prediction": pred_text.strip(),
                    "normalized_prediction": norm_pred,
                }
            )
        model.close()
        torch.cuda.empty_cache()

    write_jsonl(args.output, results)
    print(f"[symbolic-moe] Routed outputs saved to {args.output}")


if __name__ == "__main__":
    main()
