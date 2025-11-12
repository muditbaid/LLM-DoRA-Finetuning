#!/usr/bin/env python
"""Evaluate a LoRA/DoRA-adapted Llama model on the Kaggle Cyberbullying validation split.

Metrics:
- label_accuracy: accuracy over {bully, not_bully}
- type_accuracy: accuracy over {age, gender, ethnicity, religion, none}
- joint_accuracy: both label and type match
- label_macro_f1: macro-F1 for {bully, not_bully}
- type_macro_f1 (bully-only): macro-F1 over {age, gender, ethnicity, religion} on examples with gold label=bully

Usage:
  python scripts/eval_cyberbullying_metrics.py \
    --adapter saves/llama31-8b/kaggle_cyberbullying/dora/checkpoint-XXXX \
    [--base-model meta-llama/Meta-Llama-3.1-8B-Instruct] \
    [--data-file data/kaggle_cyberbullying_validation.jsonl] [--batch 8]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer


LABELS = ["bully", "not_bully"]
TYPES = ["age", "gender", "ethnicity", "religion", "none"]
PAIR_RE = re.compile(
    r"label\s*:\s*(bully|not[_\s]?bully)\s*;\s*type\s*:\s*(age|gender|ethnicity|religion|none)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", required=True, help="Path to LoRA/DoRA adapter or checkpoint dir.")
    p.add_argument(
        "--base-model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Base model name or path (must match training base).",
    )
    p.add_argument(
        "--data-file",
        default=str(Path(__file__).resolve().parents[1] / "data" / "kaggle_cyberbullying_validation.jsonl"),
        help="Path to validation JSONL in Alpaca fields (instruction,input,output,system).",
    )
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save metrics as JSON. Use 'auto' to write to the adapter directory.",
    )
    return p.parse_args()


def build_llama3_prompt(system: str, instruction: str, user_input: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        + (system or "")
        + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        + instruction
        + "\n"
        + (user_input or "")
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def load_data(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_pred(text: str) -> Tuple[str, str]:
    text = (text or "").strip()
    m = PAIR_RE.search(text)
    if not m:
        return "", ""
    lbl = m.group(1).lower().replace(" ", "_")
    typ = m.group(2).lower()
    if lbl == "notbully":  # tolerate missing underscore
        lbl = "not_bully"
    return lbl, typ


@torch.no_grad()
def generate(model, tokenizer, prompts: List[str], max_new_tokens: int, device: str) -> List[str]:
    tok = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = tok["input_ids"].to(device)
    attn_mask = tok["attention_mask"].to(device)
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
    outs: List[str] = []
    for i in range(gen.size(0)):
        new_tokens = gen[i, input_ids.size(1) :]
        outs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outs


def main() -> None:
    args = parse_args()
    device = args.device

    print("[eval] Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"
    tokenizer.padding_side = "left"

    print("[eval] Loading model…")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device)
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id

    data = load_data(args.data_file)
    print(f"[eval] Loaded {len(data)} examples from {args.data_file}")

    gold_lbls: List[str] = []
    gold_types: List[str] = []
    prompts: List[str] = []
    for ex in data:
        # gold output is in the exact format we train with; parse it back
        gl, gt = parse_pred(ex["output"])  # expected
        gold_lbls.append(gl)
        gold_types.append(gt)
        prompts.append(build_llama3_prompt(ex.get("system", ""), ex["instruction"], ex.get("input", "")))

    preds_lbl: List[str] = []
    preds_typ: List[str] = []
    for i in range(0, len(prompts), args.batch):
        batch_prompts = prompts[i : i + args.batch]
        texts = generate(model, tokenizer, batch_prompts, args.max_new_tokens, device)
        for t in texts:
            l, ty = parse_pred(t)
            preds_lbl.append(l if l in LABELS else "")
            preds_typ.append(ty if ty in TYPES else "")
        if ((i // args.batch) + 1) % 10 == 0:
            print(f"[eval] Processed {min(i+args.batch, len(prompts))}/{len(prompts)} examples…")

    # Joint match
    joint = [int(pl == gl and pt == gt) for pl, pt, gl, gt in zip(preds_lbl, preds_typ, gold_lbls, gold_types)]
    joint_acc = sum(joint) / len(joint)

    # Label metrics
    label_acc = accuracy_score(gold_lbls, preds_lbl)
    label_f1 = f1_score(gold_lbls, preds_lbl, labels=LABELS, average="macro", zero_division=0)

    # Type metrics (all examples, expecting 'none' when not_bully)
    type_acc = accuracy_score(gold_types, preds_typ)

    # Bully-only type macro-F1
    bully_mask = [g == "bully" for g in gold_lbls]
    bully_gold = [gt for gt, m in zip(gold_types, bully_mask) if m]
    bully_pred = [pt for pt, m in zip(preds_typ, bully_mask) if m]
    if bully_gold:
        type_macro_f1 = f1_score(bully_gold, bully_pred, labels=TYPES[:-1], average="macro", zero_division=0)
    else:
        type_macro_f1 = 0.0

    label_report_dict = classification_report(
        gold_lbls, preds_lbl, labels=LABELS, digits=4, zero_division=0, output_dict=True
    )
    label_report_text = classification_report(
        gold_lbls, preds_lbl, labels=LABELS, digits=4, zero_division=0, output_dict=False
    )
    type_report_dict = classification_report(
        gold_types, preds_typ, labels=TYPES, digits=4, zero_division=0, output_dict=True
    )
    type_report_text = classification_report(
        gold_types, preds_typ, labels=TYPES, digits=4, zero_division=0, output_dict=False
    )

    print("\nKaggle Cyberbullying validation metrics")
    print(f"adapter: {args.adapter}")
    print(f"label_accuracy: {label_acc:.4f}")
    print(f"label_macro_f1: {label_f1:.4f}")
    print(f"type_accuracy: {type_acc:.4f}")
    print(f"type_macro_f1_bully_only: {type_macro_f1:.4f}")
    print(f"joint_accuracy: {joint_acc:.4f}")

    print("\nLabel report:\n")
    print(label_report_text)

    print("Type report (all examples):\n")
    print(type_report_text)

    if args.save_json:
        metrics = {
            "adapter": args.adapter,
            "data_file": args.data_file,
            "label_accuracy": label_acc,
            "label_macro_f1": label_f1,
            "type_accuracy": type_acc,
            "type_macro_f1_bully_only": type_macro_f1,
            "joint_accuracy": joint_acc,
            "label_report": label_report_dict,
            "type_report": type_report_dict,
        }
        save_path = args.save_json
        if save_path == "auto":
            save_path = Path(args.adapter) / "eval_results.json"
        else:
            save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\n[eval] Saved metrics JSON to {save_path}")


if __name__ == "__main__":
    main()
