#!/usr/bin/env python3
"""
Route samples to experts using learned profiles and return each expert's decision.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

SYMBOLIC_ROOT = Path(__file__).resolve().parent
if str(SYMBOLIC_ROOT) not in sys.path:
    sys.path.insert(0, str(SYMBOLIC_ROOT))

from langsmith_utils import (
    LangSmithManager,
    build_common_metadata,
    default_dataset_name_for_split,
    infer_split,
    parse_tags,
    root_trace_inputs,
    sample_key_for_row,
    skill_stage_outputs,
    task_label_space,
)


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
eval_mod = _load_module(SYMBOLIC_ROOT / "evaluate_outputs.py", "symbolic_moe_eval_route")

normalize_label = build_profiles_mod.normalize_label
EXPERTS = config_mod.EXPERTS
PROFILES_PATH = config_mod.PROFILES_PATH
SKILL_FIELD = config_mod.SKILL_FIELD
SKILL_VOCAB = config_mod.SKILL_VOCAB
SKILL_FILE = config_mod.SKILL_FILE
TEST_SAMPLE = config_mod.TEST_SAMPLE
read_jsonl = io_mod.read_jsonl
write_jsonl = io_mod.write_jsonl
ExpertModel = model_mod.ExpertModel
evaluate_row = eval_mod.evaluate_row

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
    parser.add_argument(
        "--langsmith-project",
        type=str,
        default=None,
        help="Optional LangSmith project override for router traces.",
    )
    parser.add_argument(
        "--langsmith-tags",
        type=str,
        default="",
        help="Optional comma-separated LangSmith tags.",
    )
    parser.add_argument(
        "--langsmith-dataset-name",
        type=str,
        default=None,
        help="Optional LangSmith dataset name used to attach reference example ids.",
    )
    args = parser.parse_args()

    samples = read_jsonl(args.input)
    profiles = load_profiles()
    skill_odds, priors = build_skill_odds(profiles)
    split = infer_split(args.input)
    ls_manager = LangSmithManager(
        project_name=args.langsmith_project,
        tags=["router:profile_logodds", f"split:{split}", *parse_tags(args.langsmith_tags)],
    )
    ls_metadata = build_common_metadata(
        skills_path=SKILL_FILE,
        profiles_path=PROFILES_PATH,
        split=split,
        cwd=Path.cwd(),
        extra={
            "router_name": "profile_logodds",
            "router_script": "route_and_predict.py",
            "alpha": ALPHA,
            "base_model": config_mod.BASE_MODEL,
            "keyword_model": getattr(config_mod, "KEYWORD_MODEL", config_mod.BASE_MODEL),
            "input_path": str(args.input),
            "output_path": str(args.output),
            "expert_names": [cfg.name for cfg in EXPERTS],
        },
    )
    dataset_name = args.langsmith_dataset_name or default_dataset_name_for_split(split)
    example_id_by_key = (
        ls_manager.dataset_example_map(dataset_name=dataset_name)
        if ls_manager.enabled and dataset_name
        else {}
    )

    assignments = defaultdict(list)  # expert_name -> list of (sample_idx, weight)
    routed_info: list[dict] = []
    results = [dict(sample) for sample in samples]
    trace_predictions = defaultdict(list)  # sample_idx -> richer trace-only prediction metadata
    root_runs: dict[int, object] = {}

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
        score_map = {}
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

            score_map[cfg.name] = weight
            if weight > 0:
                candidates.append((cfg.name, weight))
            print(f"[routing] weight for {cfg.name}: {weight}")

        candidates.sort(key=lambda x: x[1], reverse=True)
        ranked_scores = sorted(score_map.items(), key=lambda x: x[1], reverse=True)

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
                "sample_key": (sample.get("dataset"), sample.get("id")),
                "id": sample.get("id"),
                "raw_skills": raw_skills,
                "mapped_skills": mapped_skills,
                "score_map": score_map,
                "ranked_experts": [name for name, _ in ranked_scores],
                "assigned_experts": [name for name, _ in selected],
                "routing_margin": (
                    ranked_scores[0][1] - ranked_scores[1][1]
                    if len(ranked_scores) >= 2
                    else None
                ),
                "selection_policy": "relative_threshold_alpha_0.4" if mapped_skills else "positive_priors_no_skills",
            }
        )
        for name, weight in selected:
            assignments[name].append((idx, weight))

    if ls_manager.enabled:
        routed_info_by_key = {info["sample_key"]: info for info in routed_info}
        for sample_idx, sample in enumerate(samples):
            route_info = routed_info_by_key.get((sample.get("dataset"), sample.get("id")), {})
            root = ls_manager.create_root(
                name="symbolic_moe_example",
                inputs=root_trace_inputs(sample),
                metadata={
                    **ls_metadata,
                    "dataset": sample.get("dataset"),
                    "sample_id": sample.get("id"),
                    "sample_key": sample_key_for_row(sample),
                    "comparison_group": sample_key_for_row(sample),
                },
                reference_example_id=example_id_by_key.get(sample_key_for_row(sample)),
            )
            ls_manager.create_child(
                root,
                name="skill_inference",
                inputs={"input": sample.get("input", "")},
                outputs=skill_stage_outputs(
                    predicted_skills=route_info.get("mapped_skills", sample.get(SKILL_FIELD, [])),
                    skill_vote_counts=sample.get("skill_vote_counts", {}),
                    keyword_responses=sample.get("keyword_responses", []),
                    unselected_skills=sample.get("unselected_skills", []),
                    empty_skills=not bool(route_info.get("mapped_skills", sample.get(SKILL_FIELD, []))),
                ),
                metadata={"stage": "skill_inference"},
            )
            ls_manager.create_child(
                root,
                name="router_scoring",
                inputs={"predicted_skills": route_info.get("mapped_skills", sample.get(SKILL_FIELD, []))},
                outputs={
                    "score_map": route_info.get("score_map", {}),
                    "ranked_experts": route_info.get("ranked_experts", []),
                    "selected_experts": route_info.get("assigned_experts", []),
                    "routing_margin": route_info.get("routing_margin"),
                    "selection_policy": route_info.get("selection_policy"),
                },
                metadata={"stage": "router"},
            )
            root_runs[sample_idx] = root

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
            started = time.perf_counter()
            pred_text, confidence = model.predict_with_confidence(prompt)
            latency_seconds = time.perf_counter() - started
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
            trace_predictions[sample_idx].append(
                {
                    "expert": cfg.name,
                    "weight": weight,
                    "label_confidence": confidence,
                    "raw_prediction": pred_text.strip(),
                    "normalized_prediction": norm_pred,
                    "task_label_space": task_label_space(cfg.label_texts),
                    "latency_seconds": latency_seconds,
                }
            )
            if ls_manager.enabled:
                ls_manager.create_child(
                    root_runs.get(sample_idx),
                    name="expert_inference",
                    inputs={
                        "expert": cfg.name,
                        "prompt": prompt,
                        "task_label_space": task_label_space(cfg.label_texts),
                        "routed_weight": weight,
                    },
                    outputs={
                        "raw_prediction": pred_text.strip(),
                        "normalized_prediction": norm_pred,
                        "label_confidence": confidence,
                        "latency_seconds": latency_seconds,
                    },
                    metadata={
                        "stage": "expert_inference",
                        "expert": cfg.name,
                    },
                )
        model.close()
        torch.cuda.empty_cache()

    write_jsonl(args.output, results)
    print(f"[symbolic-moe] Routed outputs saved to {args.output}")

    if ls_manager.enabled:
        for sample_idx, result in enumerate(results):
            evaluation = evaluate_row(result, max_k_considered=4)
            ls_manager.create_child(
                root_runs.get(sample_idx),
                name="evaluation",
                inputs={
                    "gold_label": result.get("output", ""),
                    "dataset": result.get("dataset", ""),
                },
                outputs=evaluation,
                metadata={"stage": "evaluation"},
            )
            ls_manager.end_run(
                root_runs.get(sample_idx),
                outputs={
                    "predictions": trace_predictions.get(sample_idx, result.get("predictions", [])),
                    "evaluation": evaluation,
                },
            )


if __name__ == "__main__":
    main()
