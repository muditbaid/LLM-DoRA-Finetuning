"""
Small numeric helpers.
"""
from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def prob_to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log(1 - p)


def logit_to_prob(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))
