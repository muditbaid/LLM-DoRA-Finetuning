"""
Functions to transform raw label-token scores into union-level logits.
"""
from __future__ import annotations

import numpy as np

from .model_registry import MODEL_REGISTRY
from .constants import UNION_LABELS, BULLY_SUBTYPES, HIGH_LEVEL_LABEL
from .math_utils import prob_to_logit


def raw_scores_to_union_logits(model_name: str, raw_scores: np.ndarray) -> np.ndarray:
    """
    Args:
        raw_scores: (N, K) unnormalized log scores returned by LabelScorer.
    Returns:
        (N, L) logits aligned with UNION_LABELS (NaN when model doesn't cover that label).
    """
    cfg = MODEL_REGISTRY[model_name]
    shifted = raw_scores - np.max(raw_scores, axis=1, keepdims=True)
    probs = np.exp(shifted)
    denom = np.clip(np.sum(probs, axis=1, keepdims=True), 1e-8, None)
    probs /= denom
    union_logits = np.full((raw_scores.shape[0], len(UNION_LABELS)), np.nan, dtype=np.float32)
    for i in range(raw_scores.shape[0]):
        mapped = cfg.map_probs_to_union(probs[i])
        union_logits[i] = prob_to_logit(mapped)
    return union_logits


def enforce_bully_hierarchy(prob_matrix: np.ndarray) -> np.ndarray:
    """
    Sets bully probability to max subtype probability (if any subtype defined).
    """
    proba = prob_matrix.copy()
    if HIGH_LEVEL_LABEL not in UNION_LABELS:
        return proba
    bully_idx = UNION_LABELS.index(HIGH_LEVEL_LABEL)
    subtype_indices = [UNION_LABELS.index(lbl) for lbl in BULLY_SUBTYPES]
    subtype_probs = proba[:, subtype_indices]
    bully_prob = np.nanmax(subtype_probs, axis=1)
    proba[:, bully_idx] = bully_prob
    return proba
