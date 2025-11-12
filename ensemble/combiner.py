"""
Combine calibrated logits from multiple adapters.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .constants import UNION_LABELS
from .math_utils import sigmoid
from .transforms import enforce_bully_hierarchy


def combine_logits(
    calibrated_logits: Dict[str, np.ndarray],
    weights: Dict[str, Dict[str, float]],
) -> np.ndarray:
    models = list(calibrated_logits.keys())
    n = next(iter(calibrated_logits.values())).shape[0]
    L = len(UNION_LABELS)
    combined = np.full((n, L), np.nan, dtype=np.float32)

    for j, lbl in enumerate(UNION_LABELS):
        weighted_sum = np.zeros(n, dtype=np.float32)
        weight_tot = np.zeros(n, dtype=np.float32)
        for model in models:
            column = calibrated_logits[model][:, j]
            w = weights.get(model, {}).get(lbl, 0.0)
            if w <= 0:
                continue
            valid = ~np.isnan(column)
            weighted_sum[valid] += w * column[valid]
            weight_tot[valid] += w
        valid_final = weight_tot > 0
        combined[valid_final, j] = weighted_sum[valid_final] / weight_tot[valid_final]
    return combined


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    probs = sigmoid(logits)
    probs = enforce_bully_hierarchy(probs)
    return probs
