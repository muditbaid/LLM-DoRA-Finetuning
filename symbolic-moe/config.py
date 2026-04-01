"""
Configuration for Symbolic-MoE style routing within this project.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
KEYWORD_MODEL = BASE_MODEL

SYMBOLIC_ROOT = Path(__file__).resolve().parent
VALIDATION_POOL = SYMBOLIC_ROOT / "validation_pool.jsonl"
PROFILE_POOL = SYMBOLIC_ROOT / "profile_pool.jsonl"
PROFILE_POOL_SKILLS = SYMBOLIC_ROOT / "profile_pool_skills.jsonl"
TEST_SAMPLE = SYMBOLIC_ROOT / "test_sample_skills.jsonl"
PROFILES_PATH = SYMBOLIC_ROOT / "profiles.json"

SKILL_TAGS = ["hate", "offense", "bully", "threat", "none"]
SKILL_FIELD = "predicted_skills"
SKILL_FILE = SYMBOLIC_ROOT / "skills.txt"
if SKILL_FILE.exists():
    SKILL_VOCAB = [
        line.strip().lower()
        for line in SKILL_FILE.read_text().splitlines()
        if line.strip()
    ]
else:
    SKILL_VOCAB = []


@dataclass(frozen=True)
class ExpertConfig:
    name: str
    dataset: str
    label: str
    adapter_path: Path
    label_texts: List[str]
    normalizer: str = "simple"  # see normalize_label() in profile builder
    max_new_tokens: int = 8


EXPERTS: List[ExpertConfig] = [
    ExpertConfig(
        name="dynahate_hate",
        dataset="dynahate",
        label="hate",
        adapter_path=Path("saves/llama31-8b/dynahate/qlora"),
        label_texts=["hate", "not hate"],
    ),
    ExpertConfig(
        name="tweeteval_offense",
        dataset="tweeteval_offensive",
        label="offense",
        adapter_path=Path("saves/llama31-8b/tweeteval_offensive/qlora"),
        label_texts=["offensive", "not offensive"],
    ),
    ExpertConfig(
        name="kaggle_bully",
        dataset="kaggle_cyberbullying",
        label="bully",
        adapter_path=Path("saves/llama31-8b/kaggle_cyberbullying/qlora"),
        label_texts=[
            "label: bully; type: age",
            "label: bully; type: gender",
            "label: bully; type: ethnicity",
            "label: bully; type: religion",
            "label: bully; type: none",
            "label: not_bully; type: none",
        ],
        normalizer="bully",
        max_new_tokens=16,
    ),
    ExpertConfig(
        name="jigsaw_threat",
        dataset="jigsaw_threat",
        label="threat",
        adapter_path=Path("saves/llama31-8b/jigsaw_threat/qlora"),
        label_texts=["threat", "not threat"],
    ),
]
