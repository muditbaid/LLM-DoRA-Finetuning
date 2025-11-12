#!/usr/bin/env python
"""
Create balanced, size-capped dev splits for HateXplain and Kaggle cyberbullying.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .constants import (
    HATEXPLAIN_DEV_FULL,
    KAGGLE_DEV_FULL,
)

HATEXPLAIN_OUT = HATEXPLAIN_DEV_FULL.with_name("hatexplain_validation_balanced800.jsonl")
KAGGLE_OUT = KAGGLE_DEV_FULL.with_name("kaggle_cyberbullying_validation_balanced800.jsonl")

KAGGLE_PAIR_RE = re.compile(
    r"label\s*:\s*(bully|not[_\s]?bully)\s*;\s*type\s*:\s*(age|gender|ethnicity|religion|none)",
    re.IGNORECASE,
)


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_targets(counts: Dict[str, int], max_total: int) -> Dict[str, int]:
    total = sum(counts.values())
    if total <= max_total:
        return counts.copy()

    raw = {cls: counts[cls] / total * max_total for cls in counts}
    targets = {cls: int(raw_val) for cls, raw_val in raw.items()}
    remainder = max_total - sum(targets.values())

    def fractional(cls: str) -> float:
        return raw[cls] - targets[cls]

    ordered = sorted(counts.keys(), key=fractional, reverse=True)
    for cls in ordered:
        if remainder <= 0:
            break
        available = counts[cls] - targets[cls]
        if available <= 0:
            continue
        take = min(available, remainder, 1)
        targets[cls] += take
        remainder -= take
    return targets


def sample_rows(rows: List[dict], key_fn, max_total: int, seed: int) -> List[dict]:
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        key = key_fn(row)
        if key:
            buckets[key].append(row)

    counts = {k: len(v) for k, v in buckets.items()}
    targets = compute_targets(counts, max_total)

    rng = random.Random(seed)
    selected: List[dict] = []
    for key, bucket in buckets.items():
        rng.shuffle(bucket)
        take = min(targets.get(key, 0), len(bucket))
        selected.extend(bucket[:take])
    rng.shuffle(selected)
    return selected


def hatexplain_key(row: dict) -> str:
    return row.get("output", "").strip().lower()


def kaggle_key(row: dict) -> str:
    text = row.get("output", "")
    m = KAGGLE_PAIR_RE.search(text)
    if not m:
        return ""
    label = m.group(1).lower().replace(" ", "_")
    type_ = m.group(2).lower()
    if label != "bully":
        return "none"
    return type_


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=800, help="Max examples per dataset.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[build] Sampling HateXplain dev to {args.size} examples…")
    hate_rows = load_jsonl(HATEXPLAIN_DEV_FULL)
    hate_subset = sample_rows(hate_rows, hatexplain_key, args.size, args.seed)
    write_jsonl(HATEXPLAIN_OUT, hate_subset)
    print(f"[build] Wrote {len(hate_subset)} rows to {HATEXPLAIN_OUT}")

    print(f"[build] Sampling Kaggle dev to {args.size} examples…")
    kaggle_rows = load_jsonl(KAGGLE_DEV_FULL)
    kaggle_subset = sample_rows(kaggle_rows, kaggle_key, args.size, args.seed)
    write_jsonl(KAGGLE_OUT, kaggle_subset)
    print(f"[build] Wrote {len(kaggle_subset)} rows to {KAGGLE_OUT}")


if __name__ == "__main__":
    main()
