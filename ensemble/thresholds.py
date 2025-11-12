"""
Per-label threshold tuning.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import f1_score

from .constants import THRESHOLD_ARTIFACT, UNION_LABELS


def tune_thresholds(probs: np.ndarray, targets: np.ndarray, mask: np.ndarray, labels: List[str] | None = None) -> Dict[str, float]:
    labels = labels or UNION_LABELS
    thresholds: Dict[str, float] = {}
    for idx, lbl in enumerate(labels):
        m = mask[:, idx].astype(bool)
        p = probs[:, idx]
        y = targets[:, idx]
        valid = m & ~np.isnan(p)
        if valid.sum() == 0:
            thresholds[lbl] = 0.5
            continue
        p = p[valid]
        y = y[valid]
        unique = np.unique(np.round(p, 6))
        best_f1, best_t = -1.0, 0.5
        for t in unique:
            pred = (p >= t).astype(int)
            f1 = f1_score(y, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[lbl] = best_t
    return thresholds


def save_thresholds(thresholds: Dict[str, float], path: Path = THRESHOLD_ARTIFACT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(thresholds, indent=2))


def load_thresholds(path: Path = THRESHOLD_ARTIFACT) -> Dict[str, float]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())
