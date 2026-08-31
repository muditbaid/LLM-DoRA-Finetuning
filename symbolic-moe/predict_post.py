#!/usr/bin/env python3
"""
Consumer-facing inference runner:
post -> skill inference -> NB top-2 routing -> expert predictions -> aggregated output.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

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
route_mod = _load_module(SYMBOLIC_ROOT / "route_and_predict_nb.py", "symbolic_moe_route_predict_nb")
bp_mod = _load_module(SYMBOLIC_ROOT / "build_profiles.py", "symbolic_moe_build_profiles_predict")
skill_mod = _load_module(SYMBOLIC_ROOT / "skill_inference.py", "symbolic_moe_skill_predict")

EXPERTS = config_mod.EXPERTS
PROFILE_POOL_SKILLS = config_mod.PROFILE_POOL_SKILLS
PROFILES_PATH = config_mod.PROFILES_PATH
SKILL_VOCAB = set(config_mod.SKILL_VOCAB)
read_jsonl = io_mod.read_jsonl
write_jsonl = io_mod.write_jsonl
ExpertModel = model_mod.ExpertModel
SharedModelRuntime = model_mod.SharedModelRuntime
normalize_label = bp_mod.normalize_label

_NB_ROUTER_CACHE: dict[str, Any] | None = None


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


def load_nb_router() -> dict[str, Any]:
    global _NB_ROUTER_CACHE
    if _NB_ROUTER_CACHE is None:
        _NB_ROUTER_CACHE = route_mod.build_skill_nb_router(read_jsonl(PROFILE_POOL_SKILLS))
    return _NB_ROUTER_CACHE


def route_experts(
    skills: list[str],
    profiles: dict[str, Any],
    router: dict[str, Any],
) -> list[tuple[str, float]]:
    mapped = [skill for skill in route_mod.normalize_skills(skills) if skill in SKILL_VOCAB]
    score_map = route_mod.score_skill_nb(mapped, router)
    candidates = [
        (cfg.name, score_map.get(cfg.name, float("-inf")))
        for cfg in EXPERTS
        if cfg.name in profiles
    ]
    return route_mod.select_experts(
        candidates,
        routing_mode="skill_nb",
        alpha=route_mod.ALPHA,
        top_k=2,
    )


def build_expert_prompt_parts(label: str) -> tuple[str, str]:
    prompt = DEFAULT_PROMPTS.get(label)
    if not prompt:
        return "", ""
    return prompt["system"], prompt["instruction"]


def run_single(
    post: str,
    profiles: dict[str, Any],
    router: dict[str, Any],
    runtime: SharedModelRuntime,
    experts_by_name: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    with runtime.base_inference_context():
        annotation = skill_mod.annotate(
            runtime.model,
            runtime.tokenizer,
            post,
            args.runs,
            args.min_count,
            args.max_new_tokens,
            getattr(args, "max_input_tokens", 1536),
        )

    predicted_skills = list(annotation.get("predicted_skills", []))
    routed = route_experts(predicted_skills, profiles, router)

    output: list[dict[str, Any]] = []
    for expert_name, _weight in routed:
        cfg = next((expert for expert in EXPERTS if expert.name == expert_name), None)
        if cfg is None:
            continue

        system, instruction = build_expert_prompt_parts(cfg.label)
        model = experts_by_name[cfg.name]
        prompt = model.build_prompt(system, instruction, post)
        pred_text, confidence = model.predict_with_confidence(prompt)
        norm_pred = normalize_label(pred_text, cfg.normalizer)

        if norm_pred == cfg.label:
            output.append({"label": cfg.label, "confidence": round(float(confidence), 4)})

    output.sort(key=lambda item: item["confidence"], reverse=True)
    return {
        "post": post,
        "predicted_skills": predicted_skills,
        "routed_experts": [name for name, _weight in routed],
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
    parser.add_argument("--max-input-tokens", type=int, default=1536, help="Max prompt tokens for skill inference.")
    args = parser.parse_args()

    if not args.post and not args.input:
        raise SystemExit("Provide either --post or --input.")

    profiles = load_profiles()
    router = load_nb_router()
    runtime = SharedModelRuntime()
    experts_by_name = {}
    for cfg in EXPERTS:
        if cfg.name in profiles:
            experts_by_name[cfg.name] = ExpertModel(cfg, runtime=runtime)

    try:
        if args.post:
            result = run_single(args.post, profiles, router, runtime, experts_by_name, args)
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
            results.append(run_single(post, profiles, router, runtime, experts_by_name, args))

        out_path = args.output or (SYMBOLIC_ROOT / "predict_outputs.jsonl")
        write_jsonl(out_path, results)
        print(f"[predict] Wrote {len(results)} rows to {out_path}")
    finally:
        for model in experts_by_name.values():
            model.close()
        runtime.close()


if __name__ == "__main__":
    main()
