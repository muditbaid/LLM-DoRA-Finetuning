"""
Data loading and target/mask construction helpers for the ensemble.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .constants import (
    BULLY_SUBTYPES,
    HATEXPLAIN_DEV,
    HATEXPLAIN_LABELS,
    KAGGLE_DEV,
    UNION_LABELS,
    HIGH_LEVEL_LABEL,
    MAX_DEV_PER_DATASET,
    SAMPLE_SEED,
)

KAGGLE_PAIR_RE = re.compile(
    r"label\s*:\s*(bully|not[_\s]?bully)\s*;\s*type\s*:\s*(age|gender|ethnicity|religion|none)",
    re.IGNORECASE,
)


@dataclass
class Example:
    dataset_id: str
    instruction: str
    user_input: str
    system: str
    gold_output: str


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_kaggle_output(text: str) -> Tuple[str, str]:
    text = (text or "").strip()
    m = KAGGLE_PAIR_RE.search(text)
    if not m:
        return "", ""
    label = m.group(1).lower().replace(" ", "_")
    type_ = m.group(2).lower()
    if label == "notbully":
        label = "not_bully"
    return label, type_


def load_dev_examples() -> List[Example]:
    rows: List[Example] = []
    for dataset_id, path in [("hatexplain", HATEXPLAIN_DEV), ("kaggle", KAGGLE_DEV)]:
        dataset_rows = _load_jsonl(path)
        dataset_rows = _balanced_sample(dataset_rows, dataset_id, MAX_DEV_PER_DATASET)
        for row in dataset_rows:
            rows.append(
                Example(
                    dataset_id=dataset_id,
                    instruction=row.get("instruction", ""),
                    user_input=row.get("input", ""),
                    system=row.get("system", ""),
                    gold_output=row.get("output", ""),
                )
            )
    return rows


def build_targets_and_mask(examples: List[Example]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        targets: shape (N, L) binary matrix.
        mask: shape (N, L) boolean mask (1 if label annotated for the example).
    """
    n = len(examples)
    l = len(UNION_LABELS)
    targets = np.zeros((n, l), dtype=np.float32)
    mask = np.zeros((n, l), dtype=np.float32)

    idx_map: Dict[str, int] = {lbl: i for i, lbl in enumerate(UNION_LABELS)}

    for i, ex in enumerate(examples):
        if ex.dataset_id == "hatexplain":
            label = ex.gold_output.strip().lower()
            if label not in HATEXPLAIN_LABELS:
                continue
            j = idx_map[label]
            targets[i, j] = 1.0
            mask[i, j] = 1.0
        elif ex.dataset_id == "kaggle":
            label, type_ = _parse_kaggle_output(ex.gold_output)
            if not label:
                continue
            bully_idx = idx_map[HIGH_LEVEL_LABEL]
            mask[i, bully_idx] = 1.0
            targets[i, bully_idx] = 1.0 if label == "bully" else 0.0

            if label == "bully" and type_ in BULLY_SUBTYPES:
                for subtype in BULLY_SUBTYPES:
                    subtype_idx = idx_map[subtype]
                    mask[i, subtype_idx] = 1.0
                    targets[i, subtype_idx] = 1.0 if subtype == type_ else 0.0
        else:
            raise ValueError(f"Unknown dataset_id={ex.dataset_id}")

    return targets, mask


def split_by_dataset(examples: List[Example]) -> Dict[str, List[int]]:
    """
    Returns a mapping dataset_id -> list of indices for that dataset.
    """
    mapping: Dict[str, List[int]] = {}
    for idx, ex in enumerate(examples):
        mapping.setdefault(ex.dataset_id, []).append(idx)
    return mapping


def _balanced_sample(rows: List[dict], dataset_id: str, max_size: int) -> List[dict]:
    if len(rows) <= max_size or max_size <= 0:
        return rows
    rng = random.Random(SAMPLE_SEED)
    buckets: Dict[str, List[dict]] = {}
    for row in rows:
        if dataset_id == "hatexplain":
            key = row.get("output", "").strip().lower()
        else:
            label, subtype = _parse_kaggle_output(row.get("output", ""))
            key = subtype if label == "bully" and subtype in BULLY_SUBTYPES else "none"
        if not key:
            continue
        buckets.setdefault(key, []).append(row)

    counts = {k: len(v) for k, v in buckets.items()}
    total = sum(counts.values())
    if total == 0:
        return rows[:max_size]
    raw = {cls: counts[cls] / total * max_size for cls in counts}
    targets = {cls: int(raw_val) for cls, raw_val in raw.items()}
    remainder = max_size - sum(targets.values())
    ordered = sorted(counts.keys(), key=lambda c: raw[c] - targets[c], reverse=True)
    for cls in ordered:
        if remainder <= 0:
            break
        available = counts[cls] - targets[cls]
        if available <= 0:
            continue
        targets[cls] += min(available, remainder)
        remainder -= min(available, remainder)

    sampled: List[dict] = []
    for key, bucket in buckets.items():
        rng.shuffle(bucket)
        take = min(targets.get(key, 0), len(bucket))
        sampled.extend(bucket[:take])
    rng.shuffle(sampled)
    return sampled
