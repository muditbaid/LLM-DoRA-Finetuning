#!/usr/bin/env python
"""Convert the HateXplain classification dataset into Alpaca-style JSONL files.

The resulting files can be consumed directly by LLaMA-Factory for SFT.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List

from datasets import DatasetDict, load_dataset

DEFAULT_SYSTEM_PROMPT = (
    "You are a content moderation assistant. Classify each social media post and respond with exactly one label — "
    "hatespeech, offensive, or normal — in lowercase with no additional words."
)
DEFAULT_INSTRUCTION = """The user input is a social media post. Your task is to classify the post into "hate speech", "offensive" or "normal".

Examples
Text: <user> <user> well she ’ muslim so of course she hates the jews jew hatred is one of the defining feature of islam
Label: hatespeech

Text: an these nigger biches look like godzilla nasty
Label: hatespeech

Text: <user> <user> why are you repeating yourself are you a little retarded
Label: offensive

Text: <user> lol not me i don ’ t deal with bms not outside children ’ ss i leave that ghetto mess for everybody else 😂 😂 😂
Label: offensive

Text: <user> men can not be raped can not be abused that why they call it violence against women and children because men are always the perpetrators women come first even before children
Label: normal

Text: <user> you are missing an essential premise whether any of those immigrants committed any crimes dumbass
Label: normal

Target post:
{post}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory where the converted JSONL files will be stored.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="hatexplain",
        help="Prefix for generated files, e.g., <prefix>_<split>.jsonl.",
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
    parser.add_argument(
        "--prompt-mode",
        choices=("both", "system_only", "instruction_only"),
        default="both",
        help=(
            "Where to place the task prompt. "
            "'system_only' (default) keeps the system prompt and clears the instruction, "
            "'instruction_only' keeps the instruction and clears the system prompt, "
            "and 'both' preserves both fields."
        ),
    )
    parser.add_argument(
        "--use-rationales",
        action="store_true",
        help="Aggregate annotator rationales and include them in the input context.",
    )
    parser.add_argument(
        "--rationale-agg",
        choices=("any", "majority", "intersection"),
        default="majority",
        help="How to aggregate multiple annotators' rationale masks.",
    )
    parser.add_argument(
        "--rationale-format",
        choices=("inline", "suffix"),
        default="inline",
        help="How to present rationales: inline markers or a suffix span list.",
    )
    parser.add_argument(
        "--rationale-open",
        default="<rationale>",
        help="Opening marker when using inline rationale formatting.",
    )
    parser.add_argument(
        "--rationale-close",
        default="</rationale>",
        help="Closing marker when using inline rationale formatting.",
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
        output_path = args.output_dir / f"{args.output_prefix}_{split}.jsonl"

        instruction_template = args.instruction
        system_template = args.system_prompt
        use_instruction = args.prompt_mode != "system_only"
        use_system = args.prompt_mode != "instruction_only"
        if use_instruction:
            instruction_template = instruction_template.strip()
        system_text = system_template.strip() if use_system else ""

        with output_path.open("w", encoding="utf-8") as writer:
            for example in data:
                label_id = select_majority_label(example["annotators"]["label"])
                label_text = label_feature.int2str(label_id)
                tokens: List[str] = list(example["post_tokens"])  # list[str]
                post_text = " ".join(tokens).strip()
                if not post_text:
                    continue
                # Optionally embed rationales
                if args.use_rationales and example.get("rationales"):
                    masks: List[List[int]] = [list(m) for m in example["rationales"]]
                    L = len(tokens)
                    counts = [0] * L
                    for m in masks:
                        for i, v in enumerate(m[:L]):
                            if v:
                                counts[i] += 1
                    n = max(1, len(masks))
                    if args.rationale_agg == "any":
                        agg = [c > 0 for c in counts]
                    elif args.rationale_agg == "intersection":
                        agg = [c >= n for c in counts]
                    else:  # majority
                        threshold = (n + 1) // 2
                        agg = [c >= threshold for c in counts]

                    if args.rationale_format == "inline":
                        buf: List[str] = []
                        in_span = False
                        for tok, flag in zip(tokens, agg):
                            if flag and not in_span:
                                buf.append(args.rationale_open)
                                in_span = True
                            if not flag and in_span:
                                buf.append(args.rationale_close)
                                in_span = False
                            buf.append(tok)
                        if in_span:
                            buf.append(args.rationale_close)
                        post_text = " ".join(buf)
                    else:  # suffix span list
                        spans: List[str] = []
                        cur: List[str] = []
                        for tok, flag in zip(tokens, agg):
                            if flag:
                                cur.append(tok)
                            elif cur:
                                spans.append(" ".join(cur))
                                cur = []
                        if cur:
                            spans.append(" ".join(cur))
                        if spans:
                            span_hint = " | ".join(spans[:10])
                            post_text = f"{post_text}\n\nRationale hints: {span_hint}"
                instruction_value = (
                    instruction_template.format(post=post_text) if use_instruction else ""
                )
                record = {
                    "instruction": instruction_value,
                    "input": "" if use_instruction else post_text,
                    "output": label_text,
                    "system": system_text,
                }
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Wrote {data.num_rows} samples to {output_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
