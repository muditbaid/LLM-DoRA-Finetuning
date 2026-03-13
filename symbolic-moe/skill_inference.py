#!/usr/bin/env python3
"""
Run the keyword LLM multiple times per sample and keep only confident skills.
"""
from __future__ import annotations

import argparse
import importlib.util
from collections import Counter
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from skill_parsing import parse_skills_from_text

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


config_mod = _load_module(SYMBOLIC_ROOT / "config.py", "symbolic_moe_config_skill")
io_mod = _load_module(SYMBOLIC_ROOT / "io_utils.py", "symbolic_moe_io_skill")

KEYWORD_MODEL = config_mod.KEYWORD_MODEL
SKILL_VOCAB = config_mod.SKILL_VOCAB
read_jsonl = io_mod.read_jsonl
write_jsonl = io_mod.write_jsonl

SKILL_LIST_TEXT = "\n".join(SKILL_VOCAB) if SKILL_VOCAB else ""

PROMPT_TEMPLATE = """You are tagging which conceptual skills are expressed in a social media post for hate/offense/bullying/threat detection.

Available skills (you may ONLY choose from this list and MUST copy each name EXACTLY as written):
{skills}

Instructions:
- Select between 0 and 5 skills that are clearly demonstrated in the post.
- If a concept in the post matches a skill, output that skill’s exact name.
- If you are uncertain whether a skill applies, DO NOT select it.
- Do NOT invent new skills, synonyms, or variations of the names.
- Do NOT explain your reasoning or add any extra text.

Output format (MUST follow exactly):

If one or more skills apply:
Skills: ["skill_one","skill_two"]

If no skills apply:
Skills: []

POST:
{post}
"""

def load_model(use_quantization: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(KEYWORD_MODEL, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    q_config = None
    if use_quantization and torch.cuda.is_available():
        q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        KEYWORD_MODEL,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
        quantization_config=q_config,
    )
    model.eval()
    return model, tokenizer


def parse_skills(text: str) -> List[str]:
    return parse_skills_from_text(text, SKILL_VOCAB)


def annotate(model, tokenizer, post: str, runs: int, min_count: int, max_new_tokens: int):
    counter: Counter[str] = Counter()
    responses: List[str] = []
    for _ in range(runs):
        prompt = PROMPT_TEMPLATE.format(skills=SKILL_LIST_TEXT, post=post.strip())
        messages = [
            {"role": "system", "content": "You are a careful tagging assistant."},
            {"role": "user", "content": prompt},
        ]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
        }
        with torch.inference_mode():
            output_ids = model.generate(**gen_kwargs)
        generated = output_ids[0, input_ids.shape[-1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        responses.append(text)
        for skill in parse_skills(text):
            counter[skill] += 1

    confident = [skill for skill, count in counter.items() if count >= min_count]
    return {"predicted_skills": confident, "keyword_responses": responses}


def main():
    parser = argparse.ArgumentParser(description="Infer confident skills for each sample.")
    parser.add_argument("--input", type=Path, default=Path("symbolic-moe/validation_pool.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("symbolic-moe/validation_pool_skills.jsonl"))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    samples = read_jsonl(args.input)
    model, tokenizer = load_model()
    annotated = []
    for rec in tqdm(samples, desc="Inferring skills"):
        post = rec.get("input") or ""
        annotation = annotate(model, tokenizer, post, args.runs, args.min_count, args.max_new_tokens)
        new_row = dict(rec)
        new_row.update(annotation)
        annotated.append(new_row)

    write_jsonl(args.output, annotated)
    print(f"[symbolic-moe] Wrote {len(annotated)} annotated rows to {args.output}")


if __name__ == "__main__":
    main()
