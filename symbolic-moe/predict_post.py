#!/usr/bin/env python3
"""
Consumer-facing inference runner:
post -> skill inference -> routing -> expert predictions -> aggregated output.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

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


config_mod = _load_module(SYMBOLIC_ROOT / "config.py", "symbolic_moe_config_predict")
io_mod = _load_module(SYMBOLIC_ROOT / "io_utils.py", "symbolic_moe_io_predict")
model_mod = _load_module(SYMBOLIC_ROOT / "model_utils.py", "symbolic_moe_model_predict")
route_mod = _load_module(SYMBOLIC_ROOT / "route_and_predict.py", "symbolic_moe_route_predict")
bp_mod = _load_module(SYMBOLIC_ROOT / "build_profiles.py", "symbolic_moe_build_profiles_predict")
skill_mod = _load_module(SYMBOLIC_ROOT / "skill_inference.py", "symbolic_moe_skill_predict")

EXPERTS = config_mod.EXPERTS
PROFILES_PATH = config_mod.PROFILES_PATH
SKILL_VOCAB = set(config_mod.SKILL_VOCAB)
read_jsonl = io_mod.read_jsonl
write_jsonl = io_mod.write_jsonl
ExpertModel = model_mod.ExpertModel
SharedModelRuntime = model_mod.SharedModelRuntime
normalize_label = bp_mod.normalize_label
build_skill_odds = route_mod.build_skill_odds


DEFAULT_PROMPTS = {
    "hate": {
        "system": "Strictly respond only with the label: 'hate' or 'not hate'.",
        "instruction": "You are a helpful Assistant. Your task is to classify the social media post as hate or not hate. Post:",
    },
    "offense": {
        "system": "Strictly respond only with the label: 'offensive' or 'not offensive'.",
        "instruction": "You are a helpful Assistant. Your task is to classify the social media post as offensive or not offensive. Post:",
    },
    "threat": {
        "system": "Strictly respond only with the label: 'threat' or 'not threat'.",
        "instruction": "You are a helpful Assistant. Your task is to classify the social media post as threat or not threat. Post:",
    },
    "bully": {
        "system": "You review social media posts for bullying. Reply with a single line in the format: label: bully|not_bully; type: age|gender|ethnicity|religion|none. Use type: none whenever the post is not bullying.",
        "instruction": "",
    },
}


def load_profiles() -> dict[str, Any]:
    with PROFILES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def mapped_skills(skills: list[str]) -> list[str]:
    out: list[str] = []
    for skill in skills:
        norm = str(skill).strip().lower()
        if norm in SKILL_VOCAB:
            out.append(norm)
    return out


def route_experts(skills: list[str], profiles: dict[str, Any]) -> list[tuple[str, float]]:
    odds, priors = build_skill_odds(profiles)
    mapped = mapped_skills(skills)
    candidates: list[tuple[str, float]] = []

    for cfg in EXPERTS:
        if cfg.name not in profiles:
            continue
        prior = priors.get(cfg.name, 0.0)
        if mapped:
            local = sum(odds.get(cfg.name, {}).get(skill, 0.0) for skill in mapped)
            weight = prior + local
        else:
            # No skills inferred: fallback to prior-only routing
            weight = prior
        if weight > 0:
            candidates.append((cfg.name, weight))

    candidates.sort(key=lambda x: x[1], reverse=True)

    if mapped:
        if not candidates:
            return []
        max_w = candidates[0][1]
        alpha = 0.4
        selected = [(n, w) for n, w in candidates if w >= alpha * max_w and w > 0]
        return selected if selected else [candidates[0]]

    # For empty-skill case, keep current routing behavior.
    return candidates


def build_expert_prompt_parts(label: str) -> tuple[str, str]:
    prompt = DEFAULT_PROMPTS.get(label)
    if not prompt:
        return "", ""
    return prompt["system"], prompt["instruction"]


def run_single(
    post: str,
    profiles: dict[str, Any],
    runtime: SharedModelRuntime,
    experts_by_name: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    # Run keyword/skill inference on base model (no active adapter influence).
    with runtime.base_inference_context():
        annotation = skill_mod.annotate(
            runtime.model,
            runtime.tokenizer,
            post,
            args.runs,
            args.min_count,
            args.max_new_tokens,
        )
    predicted_skills = annotation.get("predicted_skills", [])
    routed = route_experts(predicted_skills, profiles)

    output: list[dict[str, Any]] = []
    for expert_name, _weight in routed:
        cfg = next((e for e in EXPERTS if e.name == expert_name), None)
        if cfg is None:
            continue

        system, instruction = build_expert_prompt_parts(cfg.label)
        model = experts_by_name[cfg.name]
        prompt = model.build_prompt(system, instruction, post)
        pred_text, confidence = model.predict_with_confidence(prompt)
        norm_pred = normalize_label(pred_text, cfg.normalizer)

        # v1 aggregation rule: include only positive predictions for that expert label.
        if norm_pred == cfg.label:
            output.append({"label": cfg.label, "confidence": round(float(confidence), 4)})

    output.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "post": post,
        "predicted_skills": predicted_skills,
        "output": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Symbolic-MoE on a single post or JSONL batch.")
    parser.add_argument("--post", type=str, default=None, help="Single post text to classify.")
    parser.add_argument("--input", type=Path, default=None, help="Optional JSONL input with an 'input' field.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON/JSONL path.")
    parser.add_argument("--runs", type=int, default=5, help="Skill inference runs per post.")
    parser.add_argument("--min-count", type=int, default=2, help="Min count threshold for skill confidence.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens for skill inference.")
    args = parser.parse_args()

    if not args.post and not args.input:
        raise SystemExit("Provide either --post or --input.")

    profiles = load_profiles()

    runtime = SharedModelRuntime()
    experts_by_name = {}
    for cfg in EXPERTS:
        if cfg.name in profiles:
            experts_by_name[cfg.name] = ExpertModel(cfg, runtime=runtime)

    try:
        if args.post:
            result = run_single(
                args.post,
                profiles,
                runtime,
                experts_by_name,
                args,
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        rows = read_jsonl(args.input)
        results: list[dict[str, Any]] = []
        for row in rows:
            post = str(row.get("input") or row.get("post") or "").strip()
            if not post:
                continue
            results.append(
                run_single(
                    post,
                    profiles,
                    runtime,
                    experts_by_name,
                    args,
                )
            )

        out_path = args.output or (SYMBOLIC_ROOT / "predict_outputs.jsonl")
        write_jsonl(out_path, results)
        print(f"[predict] Wrote {len(results)} rows to {out_path}")
    finally:
        for model in experts_by_name.values():
            model.close()
        runtime.close()


if __name__ == "__main__":
    main()
