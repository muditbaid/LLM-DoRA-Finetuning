"""
Masked multi-label metrics.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    label_ranking_average_precision_score,
)


def masked_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    idx = mask.astype(bool).ravel()
    if idx.sum() == 0:
        return {}
    y_flat = y_true.ravel()[idx]
    pred_flat = y_pred.ravel()[idx]
    prob_flat = probs.ravel()[idx]

    out: Dict[str, float] = {}
    out["f1_micro_masked"] = f1_score(y_flat, pred_flat, average="micro", zero_division=0)
    out["f1_macro_masked"] = f1_score(y_flat, pred_flat, average="macro", zero_division=0)
    out["hamming_loss_masked"] = hamming_loss(y_flat, pred_flat)

    sample_j, sample_f1 = [], []
    for i in range(y_true.shape[0]):
        m = mask[i].astype(bool)
        if not np.any(m):
            continue
        sample_j.append(jaccard_score(y_true[i, m], y_pred[i, m], average="binary", zero_division=0))
        sample_f1.append(f1_score(y_true[i, m], y_pred[i, m], average="binary", zero_division=0))
    out["jaccard_samples_masked"] = float(np.mean(sample_j)) if sample_j else 0.0
    out["f1_samples_masked"] = float(np.mean(sample_f1)) if sample_f1 else 0.0

    per_label_ap = []
    for j in range(y_true.shape[1]):
        m = mask[:, j].astype(bool)
        if m.sum() < 2 or len(np.unique(y_true[m, j])) < 2:
            continue
        per_label_ap.append(average_precision_score(y_true[m, j], probs[m, j]))
    out["auprc_macro_masked"] = float(np.mean(per_label_ap)) if per_label_ap else 0.0

    try:
        lrap = label_ranking_average_precision_score(y_true * mask, probs)
    except ValueError:
        lrap = 0.0
    out["lrap_masked"] = float(lrap)
    return out
