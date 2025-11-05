#!/usr/bin/env python
"""Convert the HateXplain classification dataset into Alpaca-style JSONL files.

The resulting files can be consumed directly by LLaMA-Factory for SFT.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from datasets import DatasetDict, load_dataset

DEFAULT_SYSTEM_PROMPT = (
    "You are a content moderation assistant. "
    "Label each post as hatespeech, offensive, or normal according to the HateXplain policy."
)
DEFAULT_INSTRUCTION = (
    "Classify the following social media post. "
    "Respond with exactly one label: hatespeech, offensive, or normal."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory where the converted JSONL files will be stored.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "validation"),
        help="Dataset splits to export. Each split will produce a <split>.jsonl file.",
    )
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="Instruction text to include with every example.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to include with every example.",
    )
    return parser.parse_args()


def select_majority_label(labels: Iterable[int]) -> int:
    """Return the majority-vote label id, breaking ties by earliest occurrence."""
    labels = list(labels)
    if not labels:
        raise ValueError("Encountered example without annotator labels.")
    counts = Counter(labels)
    max_count = max(counts.values())
    for label in labels:
        if counts[label] == max_count:
            return label
    # The for-loop should always return. This fallback keeps mypy happy.
    return labels[0]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset: DatasetDict = load_dataset("hatexplain")
    for split in args.splits:
        if split not in dataset:
            raise KeyError(f"Split '{split}' not available in HateXplain.")
        data = dataset[split]
        label_feature = data.features["annotators"]["label"].feature
        output_path = args.output_dir / f"hatexplain_{split}.jsonl"

        with output_path.open("w", encoding="utf-8") as writer:
            for example in data:
                label_id = select_majority_label(example["annotators"]["label"])
                label_text = label_feature.int2str(label_id)
                post_text = " ".join(example["post_tokens"]).strip()
                if not post_text:
                    continue
                record = {
                    "instruction": args.instruction,
                    "input": post_text,
                    "output": label_text,
                    "system": args.system_prompt,
                }
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Wrote {data.num_rows} samples to {output_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
