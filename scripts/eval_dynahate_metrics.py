#!/usr/bin/env python
"""Evaluate QLoRA adapters on the Dynahate binary classification task with Accuracy and Macro-F1."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer


LABELS = ["hate", "not hate"]
LABEL_PATTERN = re.compile(r"(?:^|\b)(not\s*hate|hate)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        required=True,
        nargs="+",
        help="One or more adapter checkpoints/directories to evaluate.",
    )
    parser.add_argument(
        "--base-model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Base model path or Hugging Face Hub ID.",
    )
    parser.add_argument(
        "--data-file",
        nargs="+",
        default=[str(Path(__file__).resolve().parents[1] / "data" / "dynahate_dev.jsonl")],
        help="Dynahate JSONL files (Alpaca format).",
    )
    parser.add_argument("--batch", type=int, default=8, help="Batch size for greedy decoding.")
    parser.add_argument("--cutoff-len", type=int, default=1024, help="Prompt truncation length.")
    parser.add_argument("--max-new-tokens", type=int, default=4, help="Number of tokens to decode per prompt.")
    parser.add_argument(
        "--mode",
        choices=("generate", "score"),
        default="generate",
        help="generate = greedy decode label text; score = compare log-probs for each label.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (e.g., cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Disallow model/tokenizer downloads when loading checkpoints.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to dump metrics as JSON. Use 'auto' to store next to each adapter.",
    )
    return parser.parse_args()


def build_llama3_prompt(system: str, instruction: str, user_input: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        + (system or "")
        + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        + (instruction or "")
        + ("\n" if instruction and user_input else "")
        + (user_input or "")
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def load_data(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_label(text: str) -> str:
    match = LABEL_PATTERN.search((text or "").strip())
    if not match:
        return ""
    return "not hate" if match.group(1).lower().startswith("not") else "hate"


def predict(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    max_new_tokens: int,
    device: str,
    cutoff_len: int,
) -> List[str]:
    enc = tokenizer(
        list(prompts),
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=cutoff_len,
    )
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    preds: List[str] = []
    for idx in range(outputs.size(0)):
        new_tokens = outputs[idx, input_ids.size(1) :]
        preds.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return preds


def score_example(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    label_token_ids: Sequence[Tuple[str, List[int]]],
    device: str,
    cutoff_len: int,
) -> str:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cutoff_len)
    prompt_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    prompt_len = prompt_ids.size(1)

    best_label = label_token_ids[0][0]
    best_score = -float("inf")

    for label_name, token_ids in label_token_ids:
        label_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
        ids = torch.cat([prompt_ids, label_tensor], dim=1)
        mask = torch.cat([attn_mask, torch.ones_like(label_tensor)], dim=1)
        outputs = model(input_ids=ids, attention_mask=mask)
        log_probs = torch.log_softmax(outputs.logits, dim=-1)

        score = 0.0
        for offset, token_id in enumerate(token_ids):
            pos = prompt_len - 1 + offset
            score += float(log_probs[0, pos, token_id].item())

        if score > best_score:
            best_score = score
            best_label = label_name

    return best_label


def prepare_label_token_ids(tokenizer: AutoTokenizer) -> List[Tuple[str, List[int]]]:
    label_ids: List[Tuple[str, List[int]]] = []
    for label in LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            ids = ids + [tokenizer.eos_token_id]
        label_ids.append((label, ids))
    return label_ids


def evaluate_split(
    data: Sequence[dict],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    label_token_ids: Sequence[Tuple[str, List[int]]],
) -> Tuple[List[str], List[str]]:
    golds = [(ex.get("output", "") or "").strip().lower() for ex in data]
    preds: List[str] = []

    if args.mode == "generate":
        for start in range(0, len(data), args.batch):
            chunk = data[start : start + args.batch]
            prompts = [
                build_llama3_prompt(ex.get("system", ""), ex.get("instruction", ""), ex.get("input", ""))
                for ex in chunk
            ]
            raw_preds = predict(model, tokenizer, prompts, args.max_new_tokens, args.device, args.cutoff_len)
            preds.extend(normalize_label(text) for text in raw_preds)
    else:
        for ex in data:
            prompt = build_llama3_prompt(ex.get("system", ""), ex.get("instruction", ""), ex.get("input", ""))
            label = score_example(model, tokenizer, prompt, label_token_ids, args.device, args.cutoff_len)
            preds.append(label)

    preds = [p if p in LABELS else "__invalid__" for p in preds]
    return golds, preds


def save_metrics(adapter: str, args: argparse.Namespace, payload: Dict) -> None:
    if not args.save_json:
        return
    if args.save_json == "auto":
        base = Path(adapter)
        if base.is_file():
            base = base.parent
        path = base / "dynahate_eval_metrics.json"
    else:
        path = Path(args.save_json)
        if len(args.adapter) > 1:
            path = path.with_name(f"{path.stem}_{Path(adapter).name}{path.suffix or '.json'}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[dynahate] Saved metrics to {path}")


def main() -> None:
    args = parse_args()
    device = args.device

    print("[dynahate] Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"

    label_token_ids = prepare_label_token_ids(tokenizer)
    torch_dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_available() else torch.float32

    for adapter_path in args.adapter:
        print(f"\n=== Evaluating adapter: {adapter_path} ===")
        load_device_map = None if device != "auto" else "auto"
        print("[dynahate] Loading base model…")
        t0 = perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch_dtype,
            device_map=load_device_map,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        if load_device_map is None:
            model.to(device)
        print(f"[dynahate] Base model ready in {(perf_counter()-t0):.2f}s")

        print("[dynahate] Attaching adapter…")
        t1 = perf_counter()
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        if load_device_map is None:
            model.to(device)
        model.eval()
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        print(f"[dynahate] Adapter attached in {(perf_counter()-t1):.2f}s")

        results = []
        for data_file in args.data_file:
            data = load_data(data_file)
            print(f"[dynahate] Dataset: {data_file} | examples: {len(data)} | mode: {args.mode}")
            start = perf_counter()
            golds, preds = evaluate_split(data, model, tokenizer, args, label_token_ids)
            elapsed = perf_counter() - start
            acc = accuracy_score(golds, preds)
            macro_f1 = f1_score(golds, preds, labels=LABELS, average="macro", zero_division=0)
            report = classification_report(
                golds,
                preds,
                labels=LABELS,
                digits=4,
                zero_division=0,
                output_dict=False,
            )
            print(f"Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f} | Time: {elapsed:.2f}s")
            print(report)
            results.append(
                {
                    "data_file": data_file,
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                    "examples": len(data),
                    "elapsed_sec": elapsed,
                }
            )

        save_metrics(adapter_path, args, {"adapter": adapter_path, "mode": args.mode, "results": results})

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
