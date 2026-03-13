#!/usr/bin/env python3
"""Prepare HateBenchSet for LLaMA-Factory and Symbolic-MoE pipeline tests."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

SYSTEM_PROMPT = "Strictly respond only with the label: 'hate' or 'not hate'."
INSTRUCTION = "You are a helpful Assistant. Your task is to classify the social media post as hate or not hate. Post:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="TrustAIRLab/HateBenchSet")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="hate_label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--work-dir", type=Path, default=Path("hatebench"))
    parser.add_argument("--prefix", default="hatebench")
    parser.add_argument("--smoke-size", type=int, default=8)
    return parser.parse_args()


def to_label(raw: float) -> str:
    return "hate" if float(raw) >= 0.5 else "not hate"


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1
    return total


def main() -> None:
    args = parse_args()
    if not (0 <= args.val_ratio < 1 and 0 <= args.test_ratio < 1 and args.val_ratio + args.test_ratio < 1):
        raise ValueError("val-ratio and test-ratio must be in [0,1) and sum to < 1.")

    ds = load_dataset(args.dataset_id, split=args.split)
    records = []
    for idx, ex in enumerate(ds):
        text = (ex.get(args.text_column) or "").strip()
        if not text:
            continue
        label = to_label(ex.get(args.label_column, 0.0))
        records.append(
            {
                "id": f"{args.prefix}_{idx:05d}",
                "instruction": INSTRUCTION,
                "input": text,
                "output": label,
                "system": SYSTEM_PROMPT,
            }
        )

    rng = random.Random(args.seed)
    rng.shuffle(records)
    total = len(records)
    n_val = int(total * args.val_ratio)
    n_test = int(total * args.test_ratio)
    n_train = total - n_val - n_test
    train = records[:n_train]
    val = records[n_train : n_train + n_val]
    test = records[n_train + n_val :]

    base = args.data_dir
    train_path = base / f"{args.prefix}_train.jsonl"
    val_path = base / f"{args.prefix}_val.jsonl"
    test_path = base / f"{args.prefix}_test.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    write_jsonl(test_path, test)

    def to_pool(rows: list[dict], split_name: str) -> list[dict]:
        pool = []
        for i, row in enumerate(rows):
            rec = dict(row)
            rec["dataset"] = args.prefix
            rec["label"] = rec["output"]
            rec["id"] = f"{split_name}_{args.prefix}_{i:05d}"
            pool.append(rec)
        return pool

    work = args.work_dir
    pool_all = to_pool(records, "all")
    pool_smoke = pool_all[: max(0, args.smoke_size)]
    pool_path = work / f"{args.prefix}_pool.jsonl"
    smoke_path = work / f"{args.prefix}_smoke.jsonl"
    write_jsonl(pool_path, pool_all)
    write_jsonl(smoke_path, pool_smoke)

    print(f"Wrote {len(train)} rows -> {train_path}")
    print(f"Wrote {len(val)} rows -> {val_path}")
    print(f"Wrote {len(test)} rows -> {test_path}")
    print(f"Wrote {len(pool_all)} rows -> {pool_path}")
    print(f"Wrote {len(pool_smoke)} rows -> {smoke_path}")


if __name__ == "__main__":
    main()
