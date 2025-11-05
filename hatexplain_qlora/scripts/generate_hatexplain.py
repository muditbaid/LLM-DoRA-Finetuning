#!/usr/bin/env python
"""
Quick smoke-test script: load a fine-tuned adapter and generate a single prediction.
Use the same prompt format as finetune_hx_unsloth.py to keep behavior aligned.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


FEW_SHOT_EXAMPLES = [
    {
        "text": (
            "in britain the prime minister a childless woman london mayor a moslem invader in command "
            "of london police a lesbian in command of london fire brigade a fat woman scotland first "
            "minister a childless woman"
        ),
        "label": "Hate",
    },
    {
        "text": (
            "the non partisan congressional budget office cbo has released their report on sens "
            "lindsey graham and dick durbin the dream act the report shows that their legislation "
            "would add billions to the deficit over a ten year period and lots more immigrant scum"
        ),
        "label": "Hate",
    },
    {
        "text": (
            "i live and work with many legal mexican immigrants who are great citizens and trump supporters "
            "they have no problem with deporting illegals maga"
        ),
        "label": "Not Hate",
    },
]


def build_prompt(post: str) -> str:
    post = (post or "").strip()
    sections = [
        "<system>: You are a language model trained to classify social media posts as Hate or Not Hate."
    ]
    for idx, example in enumerate(FEW_SHOT_EXAMPLES, start=1):
        sections.append(f"Example {idx}:")
        sections.append(f"<human>: {example['text']}")
        sections.append(f"<bot>: {example['label']}")
        sections.append("")
    sections.append(f"<human>: {post}")
    sections.append("<bot>:")
    return "\n".join(sections).strip()


def fetch_post(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    df = pd.read_csv(args.dataset_path)
    if args.sample_index < 0 or args.sample_index >= len(df):
        raise IndexError(f"sample_index {args.sample_index} outside dataset size {len(df)}")
    return df.iloc[args.sample_index][args.text_col]


DEFAULT_ADAPTER_DIR = Path(__file__).resolve().parent.parent / "adapter"
DEFAULT_BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def load_adapter(
    adapter_dir: str,
    base_model: str | None = None,
    load_in_4bit: bool = True,
):
    adapter_path = Path(adapter_dir)
    if base_model is None:
        base_model = DEFAULT_BASE_MODEL

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a single HX prediction using a fine-tuned adapter.")
    parser.add_argument(
        "--adapter-dir",
        default=str(DEFAULT_ADAPTER_DIR),
        help="Path to the trained adapter directory.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Optional base model path/ID; if omitted we rely on adapter_config metadata.",
    )
    parser.add_argument(
        "--dataset-path",
        default="hx.csv",
        help="CSV to sample from when --text is not provided.",
    )
    parser.add_argument("--text-col", default="text", help="Column containing the post text.")
    parser.add_argument("--sample-index", type=int, default=0, help="Row index to read from the dataset.")
    parser.add_argument("--text", default=None, help="Override text instead of pulling from the dataset.")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Generation length.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for greedy).")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling cutoff.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    post = fetch_post(args)
    prompt = build_prompt(post)

    model, tokenizer = load_adapter(
        adapter_dir=args.adapter_dir,
        base_model=args.base_model,
        load_in_4bit=True,
    )

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    model.eval()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            eos_token_id=tokenizer.eos_token_id,
        )

    completion = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    print("Prompt:\n", prompt)
    print("\nModel completion:\n", completion)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
