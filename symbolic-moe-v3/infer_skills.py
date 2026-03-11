#!/usr/bin/env python3
"""
Infer fine-grained skills for each post using a base LLaMA-3.1-8B inference run.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import FINE_GRAINED_SKILLS, KEYWORD_MODEL
from .io_utils import read_jsonl, write_jsonl

PROMPT_TEMPLATE = """You are identifying which conceptual skills are required to understand or classify a social media post for hate, offensive, bullying, and threat detection.

Available skills (choose only from this list):
{skills}

A skill should be selected ONLY if it is clearly expressed or implied by the post.
If the post is harmless, neutral, or unrelated to any of these, return an empty list.

Return your answer exactly in this format:
Skills: <skill1>,<skill2>,<skill3>

For an empty set, return:
Skills:

What conceptual skills are depicted in this post?
POST: {post}
"""


def load_model(device_map: str = "auto", use_quantization: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(KEYWORD_MODEL, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    quant_config = None
    if use_quantization and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        KEYWORD_MODEL,
        device_map=device_map,
        dtype=torch_dtype,
        trust_remote_code=True,
        quantization_config=quant_config,
    )
    model.eval()
    return model, tokenizer


def build_prompt(post: str) -> str:
    skills_str = "\n".join(FINE_GRAINED_SKILLS)
    return PROMPT_TEMPLATE.format(skills=skills_str, post=post.strip())


def parse_skills(text: str) -> List[str]:
    match = re.search(r"Skills\s*:\s*(.*)", text, flags=re.IGNORECASE)
    if not match:
        return []
    remainder = match.group(1).strip()
    if not remainder:
        return []
    skills = [s.strip().lower() for s in remainder.split(",") if s.strip()]
    deduped = list(dict.fromkeys(skills))
    normalized = []
    for skill in deduped:
        if skill in FINE_GRAINED_SKILLS:
            normalized.append(skill)
        else:
            normalized.append(skill)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer fine-grained skills for each post.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("symbolic-moe") / "test_sample.jsonl",
        help="Input JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("symbolic-moe") / "test_sample_with_skills.jsonl",
        help="Where to write augmented JSONL.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max new tokens for the keyword model.",
    )
    args = parser.parse_args()

    samples = read_jsonl(args.input)
    model, tokenizer = load_model()

    results = []
    for sample in samples:
        post_text = sample.get("input") or sample.get("instruction") or ""
        prompt = build_prompt(post_text)
        messages = [
            [
                {"role": "system", "content": "You are a careful and unbiased tagging assistant."},
                {"role": "user", "content": prompt},
            ]
        ]
        rendered = [
            tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ][0]

        inputs = tokenizer(rendered, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
        }

        with torch.inference_mode():
            output_ids = model.generate(**gen_kwargs)
        generated = output_ids[0, input_ids.shape[-1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        skills = parse_skills(text)
        sample_out = dict(sample)
        sample_out["predicted_skills"] = skills
        sample_out["keyword_response"] = text
        results.append(sample_out)

    write_jsonl(args.output, results)
    print(f"[symbolic-moe] Wrote {len(results)} rows with predicted skills to {args.output}")


if __name__ == "__main__":
    main()
