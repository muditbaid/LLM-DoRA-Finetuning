#!/usr/bin/env python3
"""
Run DeBERTa multi-label predictions on the Symbolic-MoE validation pool.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import importlib.util

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

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


io_utils = _load_module(SYMBOLIC_ROOT / "io_utils.py", "symbolic_moe_io_deberta")


LABELS = ["hate", "offense", "bully", "threat"]
DATASET_TO_LABEL = {
    "dynahate": "hate",
    "tweeteval_offensive": "offense",
    "kaggle_cyberbullying": "bully",
    "jigsaw_threat": "threat",
}
POSITIVE_BY_LABEL = {
    "hate": {"hate"},
    "offense": {"offensive"},
    "bully": {"bully"},
    "threat": {"threat"},
}


def build_gold_vector(sample: Dict[str, str]) -> List[int]:
    gold = [0] * len(LABELS)
    dataset = sample.get("dataset")
    label_name = DATASET_TO_LABEL.get(dataset)
    if not label_name:
        return gold
    raw_label = (sample.get("label") or "").strip().lower()
    if raw_label in POSITIVE_BY_LABEL[label_name]:
        gold[LABELS.index(label_name)] = 1
    return gold


def build_text(sample: Dict[str, str], text_field: str, include_instruction: bool) -> str:
    text = (sample.get(text_field) or "").strip()
    if include_instruction:
        instruction = (sample.get("instruction") or "").strip()
        if instruction:
            if text:
                return f"{instruction}\n{text}"
            return instruction
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeBERTa predictions on the validation pool.")
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/deberta-v3-base",
        help="Hugging Face model name or path.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("symbolic-moe/validation_pool.jsonl"),
        help="Input JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("symbolic-moe/deberta/predictions_validation.jsonl"),
        help="Output JSONL file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit number of samples (0 = all).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Sigmoid threshold for predicted labels.",
    )
    parser.add_argument(
        "--ignore-mismatched-sizes",
        action="store_true",
        help="Allow loading checkpoints with incompatible classifier head sizes.",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="input",
        help="JSON field to use as input text.",
    )
    parser.add_argument(
        "--include-instruction",
        action="store_true",
        help="Prepend the instruction field to the text input.",
    )
    args = parser.parse_args()

    rows = io_utils.read_jsonl(args.input)
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id={label: i for i, label in enumerate(LABELS)},
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    output_rows = []
    start_time = time.perf_counter()
    for start in tqdm(range(0, len(rows), args.batch_size), desc="deberta batches"):
        batch = rows[start : start + args.batch_size]
        texts = [build_text(row, args.text_field, args.include_instruction) for row in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().tolist()

        for row, scores in zip(batch, probs):
            predicted = [label for label, score in zip(LABELS, scores) if score >= args.threshold]
            output_row = dict(row)
            output_row["deberta_scores"] = {label: float(score) for label, score in zip(LABELS, scores)}
            output_row["deberta_predicted"] = predicted
            output_row["deberta_gold"] = build_gold_vector(row)
            output_rows.append(output_row)

    io_utils.write_jsonl(args.output, output_rows)
    elapsed = time.perf_counter() - start_time
    print(f"[deberta] wrote {len(output_rows)} rows to {args.output}")
    print(f"[deberta] total time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
