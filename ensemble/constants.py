"""
Shared constants for the ensemble tooling.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]

# Label taxonomy
HATEXPLAIN_LABELS: List[str] = ["hatespeech", "offensive", "normal"]
BULLY_SUBTYPES: List[str] = ["age", "gender", "ethnicity", "religion"]
# `bully` is derived from subtype predictions; `not_bully` is implied via 1 - p(bully)
HIGH_LEVEL_LABEL = "bully"

UNION_LABELS: List[str] = HATEXPLAIN_LABELS + [HIGH_LEVEL_LABEL] + BULLY_SUBTYPES

# Dataset definitions
HATEXPLAIN_DEV_FULL = REPO_ROOT / "data" / "hatexplain_validation.jsonl"
KAGGLE_DEV_FULL = REPO_ROOT / "data" / "kaggle_cyberbullying_validation.jsonl"

HATEXPLAIN_DEV = REPO_ROOT / "data" / "hatexplain_validation_balanced800.jsonl"
KAGGLE_DEV = REPO_ROOT / "data" / "kaggle_cyberbullying_validation_balanced800.jsonl"

MAX_DEV_PER_DATASET = 800
SAMPLE_SEED = 42

# Paths to the fine-tuned adapters + stored dev metrics (for weighting).
HATEXPLAIN_ADAPTER_DIR = REPO_ROOT / "hatexplain_qlora"
KAGGLE_ADAPTER_DIR = REPO_ROOT / "saves" / "llama31-8b" / "kaggle_cyberbullying" / "qlora"

HATEXPLAIN_METRICS_FILE = HATEXPLAIN_ADAPTER_DIR / "eval_metrics.json"
KAGGLE_METRICS_FILE = KAGGLE_ADAPTER_DIR / "eval_metrics.json"

# Default base model
DEFAULT_BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Artifact locations
ARTIFACT_DIR = REPO_ROOT / "ensemble" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATION_ARTIFACT = ARTIFACT_DIR / "calibration.json"
THRESHOLD_ARTIFACT = ARTIFACT_DIR / "thresholds.json"
WEIGHTS_ARTIFACT = ARTIFACT_DIR / "weights.json"
