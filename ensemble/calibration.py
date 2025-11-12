"""
Temperature-scaling calibration utilities.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F

from .constants import CALIBRATION_ARTIFACT, UNION_LABELS


def _fit_temperature_single(logits: np.ndarray, targets: np.ndarray) -> float:
    if logits.size < 5 or len(np.unique(targets)) < 2:
        return 1.0

    x = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)

    T = torch.nn.Parameter(torch.ones(1))
    optim = torch.optim.LBFGS([T], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optim.zero_grad()
        scaled = x / T.clamp_min(1e-6)
        loss = F.binary_cross_entropy_with_logits(scaled, y)
        loss.backward()
        return loss

    optim.step(closure)
    return float(T.detach().cpu().item())


def fit_calibration(
    model_logits: Dict[str, np.ndarray],
    targets: np.ndarray,
    mask: np.ndarray,
    labels: List[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Args:
        model_logits: mapping name -> (N, L) logits aligned with UNION_LABELS.
        targets: (N, L)
        mask: (N, L)
    """
    labels = labels or UNION_LABELS
    idx_map = {lbl: i for i, lbl in enumerate(labels)}
    results: Dict[str, Dict[str, float]] = {}

    for model_name, logits in model_logits.items():
        results[model_name] = {}
        for lbl, j in idx_map.items():
            col = logits[:, j]
            m = mask[:, j].astype(bool)
            col = col[m]
            tgt = targets[m, j]
            valid = ~np.isnan(col)
            col = col[valid]
            tgt = tgt[valid]
            if col.size == 0:
                temp = 1.0
            else:
                temp = _fit_temperature_single(col, tgt)
            results[model_name][lbl] = max(temp, 1e-3)
    return results


def save_calibration(calibration: Dict[str, Dict[str, float]], path: Path = CALIBRATION_ARTIFACT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)


def load_calibration(path: Path = CALIBRATION_ARTIFACT) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def apply_calibration(
    logits: np.ndarray,
    calibration: Dict[str, float],
    labels: Iterable[str] | None = None,
) -> np.ndarray:
    labels = list(labels) if labels else UNION_LABELS
    scaled = logits.copy()
    for idx, lbl in enumerate(labels):
        temp = calibration.get(lbl, 1.0)
        scaled[:, idx] = scaled[:, idx] / max(temp, 1e-6)
    return scaled
