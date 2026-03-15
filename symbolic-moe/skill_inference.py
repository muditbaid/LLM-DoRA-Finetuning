#!/usr/bin/env python3
"""
Run the keyword LLM multiple times per sample and keep only confident skills.
"""
from __future__ import annotations

import argparse
import gc
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

SYSTEM_PROMPT = """You are a multi-label classifier for harmful-language skills.

You must output exactly one line in this format:
Skills: ["skill_one","skill_two"]

If no skills apply, output exactly:
Skills: []

Rules:
- Choose only from the allowed skill names.
- Select 0-5 skills that best describe the post.
- Choose skills based on strength of evidence in the post.
- Rank selected skills from strongest evidence to weakest evidence.
- Only include a skill if there is clear or plausible textual evidence for it.
- Do not include weak guesses just to fill the list.

- Do not explain.
- Do not repeat the prompt.
- Do not repeat the available skills list.
- Do not output anything before or after the Skills line.
- If you echo instructions or the available skills list, the output is invalid.
"""

USER_TEMPLATE = """Allowed skills:
{skills}

Post:
{post}
"""

def load_model(quantization: str = "4bit"):
    tokenizer = AutoTokenizer.from_pretrained(KEYWORD_MODEL, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    q_config = None
    if quantization != "none":
        if not torch.cuda.is_available():
            print("[skill_inference] CUDA unavailable; disabling quantization.")
            quantization = "none"
        elif quantization == "4bit":
            q_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "8bit":
            q_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"Unsupported quantization mode: {quantization}")

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


def _is_oom_error(err: BaseException) -> bool:
    if isinstance(err, torch.OutOfMemoryError):
        return True
    if isinstance(err, RuntimeError):
        return "out of memory" in str(err).lower()
    return False


def _clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def annotate_batch(
    model,
    tokenizer,
    posts: List[str],
    runs: int,
    min_count: int,
    max_new_tokens: int,
    max_input_tokens: int,
):
    counters = [Counter() for _ in posts]
    responses_by_post: List[List[str]] = [[] for _ in posts]

    for _ in range(runs):
        rendered_batch = []
        for post in posts:
            prompt = USER_TEMPLATE.format(skills=SKILL_LIST_TEXT, post=post.strip())
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            rendered_batch.append(rendered)

        inputs = tokenizer(
            rendered_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "temperature": 0.2,
            "top_p": 0.95,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
        }
        with torch.inference_mode():
            output_ids = model.generate(**gen_kwargs)
        input_lens = attention_mask.sum(dim=1).tolist() if attention_mask is not None else [input_ids.shape[-1]] * len(posts)
        for i in range(len(posts)):
            generated = output_ids[i, int(input_lens[i]) :]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            responses_by_post[i].append(text)
            for skill in parse_skills(text):
                counters[i][skill] += 1
        del inputs, input_ids, attention_mask, output_ids
        _clear_cuda_cache()

    outputs = []
    for counter, responses in zip(counters, responses_by_post):
        confident = [skill for skill, count in counter.items() if count >= min_count]
        outputs.append(
            {
                "predicted_skills": confident,
                "keyword_responses": responses,
                "skill_vote_counts": dict(counter),
            }
        )
    return outputs


def annotate_batch_with_retry(
    model,
    tokenizer,
    posts: List[str],
    runs: int,
    min_count: int,
    max_new_tokens: int,
    max_input_tokens: int,
):
    try:
        return annotate_batch(
            model=model,
            tokenizer=tokenizer,
            posts=posts,
            runs=runs,
            min_count=min_count,
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens,
        )
    except Exception as err:  # pragma: no cover - defensive retry path
        if not _is_oom_error(err):
            raise

        _clear_cuda_cache()
        if len(posts) <= 1:
            raise RuntimeError(
                "OOM even for a single sample. Lower --max-input-tokens or --max-new-tokens, or use 4bit quantization."
            ) from err

        split = len(posts) // 2
        left = annotate_batch_with_retry(
            model=model,
            tokenizer=tokenizer,
            posts=posts[:split],
            runs=runs,
            min_count=min_count,
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens,
        )
        right = annotate_batch_with_retry(
            model=model,
            tokenizer=tokenizer,
            posts=posts[split:],
            runs=runs,
            min_count=min_count,
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens,
        )
        return left + right


def main():
    parser = argparse.ArgumentParser(description="Infer confident skills for each sample.")
    parser.add_argument("--input", type=Path, default=Path("symbolic-moe/validation_pool.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("symbolic-moe/validation_pool_skills.jsonl"))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=1536,
        help="Truncate prompt+post to this many tokens to cap memory usage.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="8bit",
        choices=("4bit", "8bit", "none"),
        help="Model loading precision mode.",
    )
    args = parser.parse_args()

    samples = read_jsonl(args.input)
    model, tokenizer = load_model(args.quantization)
    annotated = []
    for start in tqdm(range(0, len(samples), args.batch_size), desc="Inferring skills"):
        batch = samples[start : start + args.batch_size]
        posts = [(rec.get("input") or "") for rec in batch]
        annotations = annotate_batch_with_retry(
            model,
            tokenizer,
            posts,
            args.runs,
            args.min_count,
            args.max_new_tokens,
            args.max_input_tokens,
        )
        for rec, annotation in zip(batch, annotations):
            new_row = dict(rec)
            new_row.update(annotation)
            annotated.append(new_row)

    write_jsonl(args.output, annotated)
    print(f"[symbolic-moe] Wrote {len(annotated)} annotated rows to {args.output}")


if __name__ == "__main__":
    main()
