"""
Per-label weighting derived from saved dev metrics.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from .constants import (
    HATEXPLAIN_METRICS_FILE,
    KAGGLE_METRICS_FILE,
    UNION_LABELS,
    BULLY_SUBTYPES,
    HIGH_LEVEL_LABEL,
    WEIGHTS_ARTIFACT,
)


def _load_hatexplain_f1s(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    f1s: Dict[str, float] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("Accuracy", "precision")):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            label = parts[0].lower()
            try:
                f1 = float(parts[3])
            except ValueError:
                continue
            f1s[label] = f1
    return f1s


def _load_kaggle_f1s(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    f1s: Dict[str, float] = {}
    per_type = data.get("per_type", {})
    for subtype in BULLY_SUBTYPES:
        info = per_type.get(subtype)
        if info:
            f1s[subtype] = info.get("f1", 0.0)
    overall = data.get("overall", {})
    if overall:
        f1s[HIGH_LEVEL_LABEL] = overall.get("macro_f1", 0.0)
    return f1s


def build_weights() -> Dict[str, Dict[str, float]]:
    hate_f1s = _load_hatexplain_f1s(HATEXPLAIN_METRICS_FILE)
    kaggle_f1s = _load_kaggle_f1s(KAGGLE_METRICS_FILE)

    per_model = {
        "hatexplain": hate_f1s,
        "kaggle": kaggle_f1s,
    }

    weights = {model: {lbl: 0.0 for lbl in UNION_LABELS} for model in per_model}

    for lbl in UNION_LABELS:
        avail = [(model, f1s.get(lbl)) for model, f1s in per_model.items() if f1s.get(lbl) is not None]
        avail = [(m, s) for m, s in avail if s > 0]
        if not avail:
            continue
        if len(avail) == 1:
            m, _ = avail[0]
            weights[m][lbl] = 1.0
            continue
        total = sum(s for _, s in avail)
        for m, s in avail:
            w = s / total if total > 0 else 0.5
            weights[m][lbl] = float(min(0.8, max(0.2, w)))
    return weights


def save_weights(weights: Dict[str, Dict[str, float]], path: Path = WEIGHTS_ARTIFACT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights, indent=2))


def load_weights(path: Path = WEIGHTS_ARTIFACT) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())
