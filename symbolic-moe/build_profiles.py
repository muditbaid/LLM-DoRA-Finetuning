#!/usr/bin/env python3
"""
Build skill profiles for each expert adapter using the validation pool.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List

import torch

from pathlib import Path

from .config import (
    EXPERTS,
    PROFILES_PATH,
    SKILL_FIELD,
    SKILL_VOCAB,
    VALIDATION_POOL,
)
from .io_utils import read_jsonl, write_jsonl
from .model_utils import ExpertModel


def normalize_label(text: str, mode: str) -> str:
    t = " ".join(text.strip().lower().split())
    if mode == "bully":
        if "not_bully" in t or "not bully" in t:
            return "not_bully"
        if "label: bully" in t or "bully" in t:
            return "bully"
        return t
    return t


def select_records(records: List[dict], dataset: str, limit: int | None) -> List[dict]:
    dataset_records = [r for r in records if r["dataset"] == dataset]
    if limit is not None:
        return dataset_records[:limit]
    return dataset_records


def build_profiles(input_path: Path, limit: int | None = None) -> Dict[str, Dict]:
    records = read_jsonl(input_path)
    profiles = {}
    predictions_dir = PROFILES_PATH.parent / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    for expert_cfg in EXPERTS:
        expert_records = select_records(records, expert_cfg.dataset, limit)
        if not expert_records:
            continue

        print(f"[symbolic-moe] Profiling {expert_cfg.name} on {len(expert_records)} samples…")
        model = ExpertModel(expert_cfg)
        skill_scores = defaultdict(int)
        stats = defaultdict(lambda: {"correct": 0, "total": 0})
        prediction_rows = []
        total_correct = 0

        for rec in expert_records:
            prompt = model.build_prompt(rec.get("system", ""), rec.get("instruction", ""), rec.get("input", ""))
            pred_text = model.predict(prompt)
            gold_text = rec.get("output", "")
            norm_pred = normalize_label(pred_text, expert_cfg.normalizer)
            norm_gold = normalize_label(gold_text, expert_cfg.normalizer)
            raw_skills = rec.get(SKILL_FIELD)
            mapped_skills = []
            if isinstance(raw_skills, list) and raw_skills:
                mapped_skills = [
                    s.strip().lower()
                    for s in raw_skills
                    if s and s.strip().lower() in SKILL_VOCAB
                ]
            else:
                mapped_skills.append(rec.get("label", "none"))
            mapped_skills = mapped_skills or ["none"]

            is_correct = int(norm_pred == norm_gold)
            total_correct += is_correct
            delta = 1 if is_correct else -1
            for skill in mapped_skills:
                stats[skill]["total"] += 1
                stats[skill]["correct"] += is_correct
                skill_scores[skill] += delta
            prediction_rows.append(
                {
                    "id": rec.get("id"),
                    "dataset": rec.get("dataset"),
                    "skill_tag": mapped_skills,
                    "prompt": {
                        "system": rec.get("system", ""),
                        "instruction": rec.get("instruction", ""),
                        "input": rec.get("input", ""),
                    },
                    "gold": norm_gold,
                    "raw_prediction": pred_text.strip(),
                    "normalized_prediction": norm_pred,
                    "is_correct": bool(is_correct),
                }
            )

        total_seen = len(expert_records)
        accuracy = total_correct / total_seen if total_seen else 0.0
        profiles[expert_cfg.name] = {
            "dataset": expert_cfg.dataset,
            "label": expert_cfg.label,
            "skill_scores": skill_scores,
            "stats": stats,
            "total_seen": total_seen,
            "total_correct": total_correct,
            "accuracy": accuracy,
        }

        pred_path = predictions_dir / f"{expert_cfg.name}.jsonl"
        write_jsonl(pred_path, prediction_rows)
        model.close()
        torch.cuda.empty_cache()

    # convert defaultdicts to plain dicts
    serializable_profiles = {}
    for name, profile in profiles.items():
            serializable_profiles[name] = {
                "dataset": profile["dataset"],
                "label": profile["label"],
            "skill_scores": dict(profile["skill_scores"]),
            "stats": {k: dict(v) for k, v in profile["stats"].items()},
            "total_seen": profile["total_seen"],
            "total_correct": profile["total_correct"],
            "accuracy": profile["accuracy"],
        }

    PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROFILES_PATH.open("w", encoding="utf-8") as f:
        json.dump(serializable_profiles, f, indent=2, ensure_ascii=False)
    print(f"[symbolic-moe] Saved profiles → {PROFILES_PATH}")
    return serializable_profiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Symbolic-MoE expert profiles.")
    parser.add_argument(
        "--input",
        type=Path,
        default=VALIDATION_POOL,
        help="Validation pool JSONL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit per dataset (for quick tests).",
    )
    args = parser.parse_args()
    build_profiles(args.input, args.limit)
