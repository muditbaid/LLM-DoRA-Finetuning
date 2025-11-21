#!/usr/bin/env python3
"""
Run the keyword LLM multiple times per sample and keep only confident skills.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import KEYWORD_MODEL, SKILL_VOCAB
from .io_utils import read_jsonl, write_jsonl

SKILL_LIST_TEXT = "\n".join(SKILL_VOCAB) if SKILL_VOCAB else ""

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
    match = re.search(r"Skills\s*:(.*)", text, flags=re.IGNORECASE)
    if not match:
        return []
    remainder = match.group(1).strip()
    if not remainder:
        return []
    allowed = set(SKILL_VOCAB)
    skills = []
    for token in remainder.split(","):
        skill = token.strip().lower()
        if skill and skill in allowed:
            skills.append(skill)
    return skills


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
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    samples = read_jsonl(args.input)
    model, tokenizer = load_model()
    annotated = []
    for rec in samples:
        post = rec.get("input") or ""
        annotation = annotate(model, tokenizer, post, args.runs, args.min_count, args.max_new_tokens)
        new_row = dict(rec)
        new_row.update(annotation)
        annotated.append(new_row)

    write_jsonl(args.output, annotated)
    print(f"[symbolic-moe] Wrote {len(annotated)} annotated rows to {args.output}")


if __name__ == "__main__":
    main()
