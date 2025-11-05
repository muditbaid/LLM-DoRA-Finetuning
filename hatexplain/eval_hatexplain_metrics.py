#!/usr/bin/env python
"""Evaluate a LoRA-adapted Llama 3.1 model on HateXplain labels with Accuracy and Macro-F1.

Usage:
  python scripts/eval_hatexplain_metrics.py \
    --adapter saves/llama31-8b/hatexplain/qlora/checkpoint-1500 \
              saves/llama31-8b/hatexplain/qlora/checkpoint-1800 \
              saves/llama31-8b/hatexplain/qlora/checkpoint-2100 \
    --data-file data/hatexplain_validation.jsonl data/hatexplain_test.jsonl \
    --mode score --local-files-only
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
from time import perf_counter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--adapter",
        required=True,
        nargs="+",
        help="One or more LoRA adapter checkpoints to evaluate.",
    )
    p.add_argument(
        "--base-model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Base model name or path (must match training base).",
    )
    p.add_argument(
        "--data-file",
        nargs="+",
        default=[str(Path(__file__).resolve().parents[1] / "data" / "hatexplain_validation.jsonl")],
        help="One or more Alpaca-format JSONL files to evaluate.",
    )
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--cutoff-len", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--mode",
        choices=("generate", "score"),
        default="generate",
        help="generate: greedy decode label; score: pick label via log-probability.",
    )
    p.add_argument(
        "--local-files-only",
        action="store_true",
        help="Force loading models/tokenizers from local cache only.",
    )
    p.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save metrics as JSON. Use 'auto' to write to the adapter directory.",
    )
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
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int,
    device: str,
    cutoff_len: int,
) -> List[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cutoff_len,
    )
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


def score_example(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    label_token_ids: List[Tuple[str, List[int]]],
    device: str,
    cutoff_len: int,
) -> str:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=cutoff_len,
    )
    prompt_ids = enc["input_ids"].to(device)
    prompt_mask = enc["attention_mask"].to(device)
    prompt_len = prompt_ids.size(1)

    best_label = LABELS[0]
    best_score = -float("inf")

    for label_name, token_ids in label_token_ids:
        label_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
        input_ids = torch.cat([prompt_ids, label_tensor], dim=1)
        attention_mask = torch.cat([prompt_mask, torch.ones_like(label_tensor)], dim=1)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            log_probs = torch.log_softmax(outputs.logits, dim=-1)

        score = 0.0
        for idx, token_id in enumerate(token_ids):
            position = prompt_len - 1 + idx
            score += float(log_probs[0, position, token_id].item())

        if score > best_score:
            best_score = score
            best_label = label_name

    return best_label


def evaluate_split(
    data: List[dict],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    label_token_ids: List[Tuple[str, List[int]]],
) -> Tuple[List[str], List[str]]:
    golds = [ex["output"].strip().lower() for ex in data]
    preds: List[str] = []

    if args.mode == "generate":
        batch = args.batch
        for i in range(0, len(data), batch):
            chunk = data[i : i + batch]
            prompts = [
                build_llama3_prompt(ex.get("system", ""), ex["instruction"], ex.get("input", "")) for ex in chunk
            ]
            texts = predict(model, tokenizer, prompts, args.max_new_tokens, args.device, args.cutoff_len)
            preds.extend([normalize_label(t) for t in texts])
            if ((i // batch) + 1) % 10 == 0:
                print(f"[eval] Processed {min(i+batch, len(data))}/{len(data)} examples…")
    else:
        for idx, ex in enumerate(data):
            prompt = build_llama3_prompt(ex.get("system", ""), ex["instruction"], ex.get("input", ""))
            label = score_example(model, tokenizer, prompt, label_token_ids, args.device, args.cutoff_len)
            preds.append(label)
            if (idx + 1) % 200 == 0:
                print(f"[eval] Scored {idx+1}/{len(data)} examples")

    preds = [p if p in LABELS else "__invalid__" for p in preds]
    return golds, preds


def main() -> None:
    args = parse_args()
    device = args.device

    print("[eval] Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    # Ensure a valid padding token for batched evaluation
    if tokenizer.pad_token is None:
        # Align with LLaMA-Factory behavior (pad=eos)
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Fallback to unk if eos is missing
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"

    label_token_ids: List[Tuple[str, List[int]]] = []
    for label in LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            ids = ids + [tokenizer.eos_token_id]
        label_token_ids.append((label, ids))

    for adapter_path in args.adapter:
        print(f"\n=== Evaluating adapter: {adapter_path} ===")
        print("[eval] Loading base model…")
        t0 = perf_counter()
        load_device_map = None if device != "auto" else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            device_map=load_device_map,
            local_files_only=args.local_files_only,
        )
        if load_device_map is None:
            model.to(device)
        print(f"[eval] Base model loaded in {(perf_counter()-t0):.2f}s on {device}.")

        try:
            model.config.pad_token_id = tokenizer.pad_token_id
        except Exception:
            pass

        print(f"[eval] Attaching adapter…")
        t1 = perf_counter()
        model = PeftModel.from_pretrained(model, adapter_path)
        if load_device_map is None:
            model.to(device)
        model.eval()
        try:
            model.config.pad_token_id = tokenizer.pad_token_id
            if getattr(model.config, "use_cache", None) is not False:
                model.config.use_cache = True
        except Exception:
            pass
        print(f"[eval] Adapter attached in {(perf_counter()-t1):.2f}s.")

        collected = []
        for data_file in args.data_file:
            data = load_data(data_file)
            print(f"[eval] Dataset: {data_file} | examples: {len(data)} | mode: {args.mode}")
            golds, preds = evaluate_split(data, model, tokenizer, args, label_token_ids)
            acc = accuracy_score(golds, preds)
            f1_macro = f1_score(golds, preds, labels=LABELS, average="macro", zero_division=0)
            report_dict = classification_report(
                golds,
                preds,
                labels=LABELS,
                digits=4,
                zero_division=0,
                output_dict=True,
            )
            report_text = classification_report(
                golds,
                preds,
                labels=LABELS,
                digits=4,
                zero_division=0,
                output_dict=False,
            )

            print(f"Accuracy: {acc:.4f} | Macro-F1: {f1_macro:.4f}")
            print(report_text)

            collected.append(
                {
                    "data_file": data_file,
                    "accuracy": acc,
                    "macro_f1": f1_macro,
                    "classification_report": report_dict,
                }
            )

        if args.save_json:
            metrics = {
                "adapter": adapter_path,
                "mode": args.mode,
                "results": collected,
            }
            if args.save_json == "auto":
                base_path = Path(adapter_path)
                if base_path.is_file():
                    base_path = base_path.parent
                save_path = base_path / "eval_results.json"
            else:
                save_path = Path(args.save_json)
                if len(args.adapter) > 1:
                    save_path = save_path.with_name(
                        f"{save_path.stem}_{Path(adapter_path).name}{save_path.suffix or '.json'}"
                    )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"[eval] Saved metrics JSON to {save_path}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
