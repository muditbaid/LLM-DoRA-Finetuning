#!/usr/bin/env python
"""Evaluate a LoRA-adapted Llama 3.1 model on HateXplain labels with Accuracy and Macro-F1.

Usage:
  python scripts/eval_hatexplain_metrics.py \
    --adapter saves/llama31-8b/hatexplain/lora/checkpoint-3500 \
    [--base-model meta-llama/Meta-Llama-3.1-8B-Instruct] \
    [--data-file data/hatexplain_validation.jsonl] [--batch 8]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Tuple

import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", required=True, help="Path to LoRA adapter or specific checkpoint directory.")
    p.add_argument(
        "--base-model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Base model name or path (must match training base).",
    )
    p.add_argument(
        "--data-file",
        default=str(Path(__file__).resolve().parents[1] / "data" / "hatexplain_validation.jsonl"),
        help="Path to Alpaca-format validation JSONL.",
    )
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--cutoff-len", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


LABELS = ["hatespeech", "offensive", "normal"]
LABEL_PATTERN = re.compile(r"^(hatespeech|offensive|normal)", re.IGNORECASE)


def build_llama3_prompt(system: str, instruction: str, user_input: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        + system
        + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        + instruction
        + "\n"
        + user_input
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def load_data(path: str) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


@torch.no_grad()
def predict(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompts: List[str], max_new_tokens: int, device: str
) -> List[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)
    # Greedy decoding
    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Slice newly generated tokens
    gen_texts: List[str] = []
    for i in range(gen.size(0)):
        new_tokens = gen[i, input_ids.size(1) :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        gen_texts.append(text)
    return gen_texts


def normalize_label(text: str) -> str:
    m = LABEL_PATTERN.match(text.strip())
    return m.group(1).lower() if m else ""


def main() -> None:
    args = parse_args()
    device = args.device

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    # Ensure a valid padding token for batched evaluation
    if tokenizer.pad_token is None:
        # Align with LLaMA-Factory behavior (pad=eos)
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Fallback to unk if eos is missing
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32, device_map="auto"
    )
    # Keep model config consistent with tokenizer padding
    try:
        model.config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    data = load_data(args.data_file)
    golds: List[str] = [ex["output"].strip().lower() for ex in data]

    preds: List[str] = []
    batch = args.batch
    for i in range(0, len(data), batch):
        chunk = data[i : i + batch]
        prompts = [
            build_llama3_prompt(ex.get("system", ""), ex["instruction"], ex.get("input", "")) for ex in chunk
        ]
        texts = predict(model, tokenizer, prompts, args.max_new_tokens, device)
        preds.extend([normalize_label(t) for t in texts])

    # Fallback: if normalization failed, mark as wrong class to avoid inflating scores
    preds = [p if p in LABELS else "__invalid__" for p in preds]

    acc = accuracy_score(golds, preds)
    f1_macro = f1_score(golds, preds, labels=LABELS, average="macro", zero_division=0)

    print("HateXplain validation metrics (adapter:)", args.adapter)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1_macro:.4f}")
    print("\nPer-class report:\n")
    print(
        classification_report(
            golds,
            preds,
            labels=LABELS,
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
