"""
Registry describing how each adapter maps its native label space
to the union label list.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from .constants import (
    HATEXPLAIN_ADAPTER_DIR,
    HATEXPLAIN_LABELS,
    KAGGLE_ADAPTER_DIR,
    BULLY_SUBTYPES,
    UNION_LABELS,
    HIGH_LEVEL_LABEL,
)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    adapter_path: str
    label_texts: List[str]
    covered_labels: List[str]

    def map_probs_to_union(self, probs: np.ndarray) -> np.ndarray:
        """Map probability vector over label_texts -> union label probabilities (NaN when unsupported)."""
        union = np.full(len(UNION_LABELS), np.nan, dtype=np.float32)
        idx_map = {lbl: i for i, lbl in enumerate(UNION_LABELS)}
        return _MODEL_MAPPERS[self.name](probs, union, idx_map)


def _hatexplain_mapper(probs: np.ndarray, union: np.ndarray, idx_map: Dict[str, int]) -> np.ndarray:
    for lbl, p in zip(HATEXPLAIN_LABELS, probs):
        union[idx_map[lbl]] = p
    return union


def _kaggle_mapper(probs: np.ndarray, union: np.ndarray, idx_map: Dict[str, int]) -> np.ndarray:
    # Order: age, gender, ethnicity, religion, not_bully (type none)
    subtype_probs = probs[: len(BULLY_SUBTYPES)]
    not_bully_prob = probs[-1]
    bully_prob = 1.0 - not_bully_prob

    union[idx_map[HIGH_LEVEL_LABEL]] = bully_prob
    for subtype, p in zip(BULLY_SUBTYPES, subtype_probs):
        union[idx_map[subtype]] = p
    return union


MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "hatexplain": ModelConfig(
        name="hatexplain",
        adapter_path=str(HATEXPLAIN_ADAPTER_DIR / "adapter"),
        label_texts=HATEXPLAIN_LABELS,
        covered_labels=HATEXPLAIN_LABELS,
    ),
    "kaggle": ModelConfig(
        name="kaggle",
        adapter_path=str(KAGGLE_ADAPTER_DIR),
        label_texts=[
            "label: bully; type: age",
            "label: bully; type: gender",
            "label: bully; type: ethnicity",
            "label: bully; type: religion",
            "label: not_bully; type: none",
        ],
        covered_labels=[HIGH_LEVEL_LABEL] + BULLY_SUBTYPES,
    ),
}

_MODEL_MAPPERS: Dict[str, Callable[[np.ndarray, np.ndarray, Dict[str, int]], np.ndarray]] = {
    "hatexplain": _hatexplain_mapper,
    "kaggle": _kaggle_mapper,
}
