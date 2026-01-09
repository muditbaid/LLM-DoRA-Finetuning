# Symbolic-MoE Routing (v2)

This directory contains the symbolic mixture-of-experts (MoE) routing pipeline that combines several task-specific QLoRA adapters (Dynahate, Jigsaw Threat, Kaggle Cyberbullying, TweetEval Offensive).

## Configuration

- **Skill list**: long, fine-grained skill inventory.
- **Max experts per example**: no hard cap (all relevant skills may be routed).
- **Alpha**: `0.6` (routing confidence vs. expert score trade-off).
- **Global score**: disabled (no separate global expert; routing is purely skill-based).

Key scripts:

- `build_profiles.py` – builds profile vectors from per-dataset examples.
- `skill_inference.py` – infers skills for incoming texts.
- `route_and_predict.py` – routes each example to the selected experts and aggregates predictions.
- `evaluate_outputs.py` – computes accuracy and per-dataset metrics from pooled predictions.

## Evaluation Results (Validation Pool, 784 Examples)

Overall and per-dataset accuracy for this configuration:

- **Overall**: 691 / 784 correct (accuracy **0.8814**)
- **Dynahate**: 183 / 196 (accuracy **0.9337**)
- **Jigsaw Threat**: 183 / 196 (accuracy **0.9337**)
- **Kaggle Cyberbullying**: 187 / 196 (accuracy **0.9541**)
- **TweetEval Offensive**: 138 / 196 (accuracy **0.7041**)

These numbers correspond to the “long skill list, alpha=0.6, no max expert, no global score” run, using the `validation_pool*.jsonl` and `test600_metrics.txt` artifacts in this folder.
## GPT-5.1 Batch Inference (skills + multilabel)

This repo includes two helper scripts for running Batch API jobs on the validation pool.

1) Create and submit batch jobs (writes `batch_ids.txt`):
```bash
export OPENAI_API_KEY=...
python symbolic-moe/batch_create.py \
  --input symbolic-moe/validation_pool.jsonl \
  --skills-file symbolic-moe/skills.txt \
  --model gpt-5.1 \
  --max-output-tokens 120 \
  --reasoning-effort medium \
  --verbosity low
```

2) Fetch outputs, archive batch files, and postprocess:
```bash
python symbolic-moe/batch_fetch.py \
  --input symbolic-moe/validation_pool.jsonl \
  --skills-file symbolic-moe/skills.txt
```

Outputs:
- `symbolic-moe/batch_files/output_files/*.jsonl`
- `symbolic-moe/batch_files/error_files/*.jsonl`
- `symbolic-moe/validation_pool_skills_gpt.jsonl` (skills + `keyword_responses`)
- `symbolic-moe/validation_pool_multilabel_gpt.jsonl` (labels + `label_response`)
