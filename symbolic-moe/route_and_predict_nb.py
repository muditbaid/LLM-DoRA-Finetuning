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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List

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


build_profiles_mod = _load_module(SYMBOLIC_ROOT / "build_profiles.py", "symbolic_moe_build_profiles_route_nb")
config_mod = _load_module(SYMBOLIC_ROOT / "config.py", "symbolic_moe_config_route_nb")
io_mod = _load_module(SYMBOLIC_ROOT / "io_utils.py", "symbolic_moe_io_route_nb")
model_mod = _load_module(SYMBOLIC_ROOT / "model_utils.py", "symbolic_moe_model_route_nb")
eval_mod = _load_module(SYMBOLIC_ROOT / "evaluate_outputs.py", "symbolic_moe_eval_route_nb")

normalize_label = build_profiles_mod.normalize_label
EXPERTS = config_mod.EXPERTS
PROFILES_PATH = config_mod.PROFILES_PATH
PROFILE_POOL_SKILLS = getattr(config_mod, "PROFILE_POOL_SKILLS", SYMBOLIC_ROOT / "profile_pool_skills.jsonl")
SKILL_FIELD = config_mod.SKILL_FIELD
SKILL_VOCAB = config_mod.SKILL_VOCAB
SKILL_FILE = config_mod.SKILL_FILE
TEST_SAMPLE = config_mod.TEST_SAMPLE
read_jsonl = io_mod.read_jsonl
write_jsonl = io_mod.write_jsonl
ExpertModel = model_mod.ExpertModel
SharedModelRuntime = model_mod.SharedModelRuntime
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


def normalize_skills(skills: Iterable[str] | str | None) -> List[str]:
    if skills is None:
        return []
    if isinstance(skills, str):
        skills = [skills]
    cleaned: List[str] = []
    for skill in skills:
        if not isinstance(skill, str):
            continue
        norm = skill.strip().lower()
        if norm in SKILL_VOCAB:
            cleaned.append(norm)
    return cleaned


def build_skill_nb_router(train_rows: list[dict]) -> dict:
    dataset_to_expert = {cfg.dataset: cfg.name for cfg in EXPERTS}
    expert_counts: Counter[str] = Counter()
    present_counts: dict[str, Counter[str]] = {cfg.name: Counter() for cfg in EXPERTS}

    for row in train_rows:
        expert_name = dataset_to_expert.get(str(row.get("dataset", "")).strip())
        if expert_name is None:
            continue
        expert_counts[expert_name] += 1
        for skill in set(normalize_skills(row.get(SKILL_FIELD) or row.get("skill_tag"))):
            present_counts[expert_name][skill] += 1

    total_rows = sum(expert_counts.values())
    num_experts = max(len(EXPERTS), 1)
    priors: dict[str, float] = {}
    likelihoods: dict[str, dict[str, float]] = {}

    for cfg in EXPERTS:
        count = expert_counts.get(cfg.name, 0)
        priors[cfg.name] = math.log((count + 1.0) / (total_rows + num_experts))
        denom = count + 2.0
        likelihoods[cfg.name] = {
            skill: (present_counts[cfg.name].get(skill, 0) + 1.0) / denom
            for skill in SKILL_VOCAB
        }

    return {"priors": priors, "likelihoods": likelihoods}


def score_profile_logodds(skills: List[str], skill_odds: dict, priors: dict) -> dict[str, float]:
    scores: dict[str, float] = {}
    for cfg in EXPERTS:
        expert_odds = skill_odds.get(cfg.name, {})
        prior = priors.get(cfg.name, 0.0)
        if skills:
            local_score = sum(expert_odds.get(skill, 0.0) for skill in skills)
            scores[cfg.name] = prior + local_score
        else:
            scores[cfg.name] = prior
    return scores


def score_skill_nb(skills: List[str], router: dict) -> dict[str, float]:
    seen = set(skills)
    priors = router["priors"]
    likelihoods = router["likelihoods"]
    scores: dict[str, float] = {}

    for cfg in EXPERTS:
        score = priors.get(cfg.name, 0.0)
        expert_likelihoods = likelihoods.get(cfg.name, {})
        for skill in SKILL_VOCAB:
            p = min(max(expert_likelihoods.get(skill, 0.5), 1e-6), 1 - 1e-6)
            score += math.log(p if skill in seen else (1.0 - p))
        scores[cfg.name] = score

    return scores


def select_experts(candidates: list[tuple[str, float]], routing_mode: str, alpha: float, top_k: int | None) -> list[tuple[str, float]]:
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

    if top_k is not None:
        return candidates[: max(1, top_k)]

    if routing_mode == "skill_nb":
        return candidates[:1]

    positive_candidates = [c for c in candidates if c[1] > 0]
    if not positive_candidates:
        return [candidates[0]]

    max_w = positive_candidates[0][1]
    selected = [(n, w) for n, w in positive_candidates if w >= alpha * max_w]
    return selected or [positive_candidates[0]]


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
        "--routing-mode",
        choices=("profile_logodds", "skill_nb"),
        default="skill_nb",
        help="Routing scorer to use. 'profile_logodds' preserves the current profile-based gate; "
        "'skill_nb' learns a Bernoulli skill router from the current profile pool.",
    )
    parser.add_argument(
        "--router-train-input",
        type=Path,
        default=PROFILE_POOL_SKILLS,
        help="Training JSONL for the discriminative skill router.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
        help="Relative threshold fraction of max score for profile_logodds routing.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="If set, route to the top-k experts by score regardless of routing mode.",
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
    split = infer_split(args.input)
    router_name = args.routing_mode
    ls_manager = LangSmithManager(
        project_name=args.langsmith_project,
        tags=[f"router:{router_name}", f"split:{split}", *parse_tags(args.langsmith_tags)],
    )
    ls_metadata = build_common_metadata(
        skills_path=SKILL_FILE,
        profiles_path=PROFILES_PATH,
        split=split,
        cwd=Path.cwd(),
        extra={
            "router_name": router_name,
            "router_script": "route_and_predict_nb.py",
            "alpha": args.alpha,
            "top_k": args.top_k,
            "base_model": config_mod.BASE_MODEL,
            "keyword_model": getattr(config_mod, "KEYWORD_MODEL", config_mod.BASE_MODEL),
            "router_train_input": str(args.router_train_input),
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
    skill_odds = priors = None
    skill_nb_router = None
    if args.routing_mode == "profile_logodds":
        skill_odds, priors = build_skill_odds(profiles)
    else:
        train_rows = read_jsonl(args.router_train_input)
        skill_nb_router = build_skill_nb_router(train_rows)
        print(
            f"[routing] Trained Bernoulli skill router from {args.router_train_input} "
            f"on {len(train_rows)} rows."
        )

    assignments = defaultdict(list)  # expert_name -> list of (sample_idx, weight)
    routed_info: list[dict] = []
    results = [dict(sample) for sample in samples]
    trace_predictions = defaultdict(list)
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
        mapped_skills = normalize_skills(raw_skills)
        print(f"[routing] Sample {sample.get('id')} raw_skills={raw_skills} mapped={mapped_skills}")
        if args.routing_mode == "profile_logodds":
            score_map = score_profile_logodds(mapped_skills, skill_odds, priors)
        else:
            score_map = score_skill_nb(mapped_skills, skill_nb_router)

        candidates = [(cfg.name, score_map.get(cfg.name, float("-inf"))) for cfg in EXPERTS]
        for cfg in EXPERTS:
            print(f"[routing] score for {cfg.name}: {score_map.get(cfg.name, float('-inf'))}")

        ranked_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        selected = select_experts(
            candidates,
            routing_mode=args.routing_mode,
            alpha=args.alpha,
            top_k=args.top_k,
        )
        print(f"[routing] Selected {len(selected)} experts for {sample.get('id')}: {selected}")

        routed_info.append(
            {
                "sample_key": (sample.get("dataset"), sample.get("id")),
                "id": sample.get("id"),
                "raw_skills": raw_skills,
                "mapped_skills": mapped_skills,
                "score_map": score_map,
                "ranked_experts": [name for name, _ in ranked_candidates],
                "assigned_experts": [name for name, _ in selected],
                "routing_margin": (
                    ranked_candidates[0][1] - ranked_candidates[1][1]
                    if len(ranked_candidates) >= 2
                    else None
                ),
                "selection_policy": (
                    f"top_k_{args.top_k}" if args.top_k is not None else f"default_{args.routing_mode}"
                ),
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
    runtime = SharedModelRuntime()
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
    finally:
        runtime.close()

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
