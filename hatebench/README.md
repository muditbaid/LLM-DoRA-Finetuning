# HateBench Pipeline Setup

This folder contains a small setup to run the current pipeline on
`TrustAIRLab/HateBenchSet`.

## 1) Prepare data

```bash
python hatebench/prepare_hatebench.py
```

Outputs:
- `data/hatebench_train.jsonl`
- `data/hatebench_val.jsonl`
- `data/hatebench_test.jsonl`
- `hatebench/hatebench_pool.jsonl`
- `hatebench/hatebench_smoke.jsonl`

## 2) Smoke test pipeline on HateBench

```bash
python symbolic-moe/skill_inference.py \
  --input hatebench/hatebench_smoke.jsonl \
  --output hatebench/hatebench_smoke_skills.jsonl \
  --runs 1 --min-count 1 --max-new-tokens 64

python symbolic-moe/route_and_predict.py \
  --input hatebench/hatebench_smoke_skills.jsonl \
  --output hatebench/hatebench_smoke_outputs.jsonl
```

## 3) HateBench-specific evaluation (binary metrics)

```bash
python hatebench/eval_hatebench_outputs.py \
  --input hatebench/hatebench_smoke_outputs.jsonl \
  --mode expert \
  --expert-name dynahate_hate \
  --metrics-out hatebench/hatebench_smoke_binary_metrics.txt
```
