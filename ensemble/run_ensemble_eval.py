#!/usr/bin/env python
"""
End-to-end calibration, ensembling, and evaluation pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

import numpy as np

from .calibration import apply_calibration, fit_calibration, save_calibration
from .combiner import combine_logits, logits_to_probs
from .constants import (
    CALIBRATION_ARTIFACT,
    THRESHOLD_ARTIFACT,
    WEIGHTS_ARTIFACT,
    UNION_LABELS,
)
from .data_utils import build_targets_and_mask, load_dev_examples, split_by_dataset
from .metrics import masked_metrics
from .model_registry import MODEL_REGISTRY
from .scoring import LabelScorer, build_llama3_prompt
from .thresholds import save_thresholds, tune_thresholds
from .transforms import raw_scores_to_union_logits
from .weights import build_weights, save_weights


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="Override device, e.g., cuda:0 or cpu.")
    parser.add_argument("--cutoff-len", type=int, default=2048)
    parser.add_argument("--save-summary", default="ensemble/artifacts/eval_summary.json")
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N prompts.")
    return parser.parse_args()


def main():
    args = parse_args()

    print("[ensemble] Loading dev splits…")
    examples = load_dev_examples()
    targets, mask = build_targets_and_mask(examples)
    dataset_indices = split_by_dataset(examples)
    prompts = [build_llama3_prompt(ex.system, ex.instruction, ex.user_input) for ex in examples]

    model_logits = {}
    for name, cfg in MODEL_REGISTRY.items():
        print(f"[ensemble] Scoring prompts with {name} adapter…")
        scorer = LabelScorer(
            adapter_path=cfg.adapter_path,
            label_texts=cfg.label_texts,
            device=args.device,
            cutoff_len=args.cutoff_len,
        )
        raw_scores = scorer.score_prompts(prompts, progress_every=args.progress_every)
        model_logits[name] = raw_scores_to_union_logits(name, raw_scores)
        scorer.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[ensemble] Fitting per-label temperature scaling…")
    calibration = fit_calibration(model_logits, targets, mask)
    save_calibration(calibration, Path(CALIBRATION_ARTIFACT))

    calibrated = {
        name: apply_calibration(model_logits[name], calibration[name], UNION_LABELS) for name in MODEL_REGISTRY
    }

    print("[ensemble] Building label weights from dev metrics…")
    weights = build_weights()
    save_weights(weights, Path(WEIGHTS_ARTIFACT))

    print("[ensemble] Combining calibrated logits…")
    combined_logits = combine_logits(calibrated, weights)
    probs = logits_to_probs(combined_logits)

    print("[ensemble] Tuning thresholds…")
    thresholds = tune_thresholds(probs, targets, mask)
    save_thresholds(thresholds, Path(THRESHOLD_ARTIFACT))

    print("[ensemble] Computing metrics…")
    preds = np.zeros_like(probs, dtype=np.int32)
    for j, lbl in enumerate(UNION_LABELS):
        tau = thresholds.get(lbl, 0.5)
        pj = probs[:, j]
        valid = ~np.isnan(pj)
        preds[valid, j] = (pj[valid] >= tau).astype(int)

    summary = masked_metrics(targets, preds, probs, mask)
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")

    summary_path = Path(args.save_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[ensemble] Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
