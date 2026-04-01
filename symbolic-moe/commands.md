# Symbolic-MoE Commands

## Batch Skills (GPT)

```bash
export OPENAI_API_KEY=...

# 1) create + submit batches
python symbolic-moe/batch_create.py \
  --input symbolic-moe/validation_pool.jsonl \
  --skills-file symbolic-moe/skills.txt \
  --model gpt-5.1

# 2) fetch + postprocess
python symbolic-moe/batch_fetch.py \
  --input symbolic-moe/validation_pool.jsonl \
  --skills-file symbolic-moe/skills.txt
```

## Pipeline

### 1) Skill inference

```bash
LOG_TS=$(date +%Y%m%d_%H%M%S)
mkdir -p symbolic-moe/logs

nohup python symbolic-moe/skill_inference.py \
  --input symbolic-moe/profile_pool.jsonl \
  --output symbolic-moe/profile_pool_skills.jsonl \
  --batch-size 16 --max-input-tokens 1536 --quantization 4bit \
  > symbolic-moe/logs/profile_skill_inference_${LOG_TS}.log 2>&1 &

nohup python symbolic-moe/skill_inference.py \
  --input symbolic-moe/validation_pool.jsonl \
  --output symbolic-moe/validation_pool_skills.jsonl \
  > symbolic-moe/logs/val_skill_inference_${LOG_TS}.log 2>&1 &

nohup python symbolic-moe/skill_inference.py \
  --input symbolic-moe/test_pool.jsonl \
  --output symbolic-moe/test_pool_skills.jsonl \
  --runs 3 --min-count 2 --max-new-tokens 64 \
  > symbolic-moe/logs/test_skill_inference_${LOG_TS}.log 2>&1 &
```

### 2) Build profiles from profile pool skills

```bash
LOG_TS=$(date +%Y%m%d_%H%M%S)
mkdir -p symbolic-moe/logs

nohup python symbolic-moe/build_profiles.py \
  --input symbolic-moe/profile_pool_skills.jsonl \
  > symbolic-moe/logs/build_profiles_${LOG_TS}.log 2>&1 &
```

### 3) Route + predict (validation + test)

```bash
LOG_TS=$(date +%Y%m%d_%H%M%S)
mkdir -p symbolic-moe/logs

nohup python symbolic-moe/route_and_predict.py \
  --input symbolic-moe/validation_pool_skills.jsonl \
  --output symbolic-moe/validation_pool_outputs.jsonl \
  > symbolic-moe/logs/route_and_predict_${LOG_TS}.log 2>&1 &

nohup python symbolic-moe/route_and_predict.py \
  --input symbolic-moe/test_pool_skills.jsonl \
  --output symbolic-moe/test_pool_outputs.jsonl \
  > symbolic-moe/logs/route_and_predict_test_${LOG_TS}.log 2>&1 &
```
```bash
nohup bash -lc '
python symbolic-moe/route_and_predict_nb.py \
  --input symbolic-moe/validation_pool_skills.jsonl \
  --output symbolic-moe/validation_pool_outputs_skill_nb_top3.jsonl \
  --top-k 3 \
  > symbolic-moe/logs/route_and_predict_nb_top3_'"$LOG_TS"'.log 2>&1 && \
python symbolic-moe/evaluate_outputs.py \
  --input symbolic-moe/validation_pool_outputs_skill_nb_top3.jsonl \
  --metrics-out symbolic-moe/validation_metrics_nb_top3.txt \
  >> symbolic-moe/logs/route_and_predict_nb_top3_'"$LOG_TS"'.log 2>&1
' &
```
### 4) Evaluate routed outputs

```bash
LOG_TS=$(date +%Y%m%d_%H%M%S)
mkdir -p symbolic-moe/logs

nohup python symbolic-moe/evaluate_outputs.py \
  --input symbolic-moe/validation_pool_outputs.jsonl \
  --metrics-out symbolic-moe/validation_metrics_${LOG_TS}.txt \
  > symbolic-moe/logs/evaluate_validation_${LOG_TS}.log 2>&1 &

nohup python symbolic-moe/evaluate_outputs.py \
  --input symbolic-moe/test_pool_outputs.jsonl \
  --metrics-out symbolic-moe/test_metrics_${LOG_TS}.txt \
  > symbolic-moe/logs/evaluate_test_${LOG_TS}.log 2>&1 &
```

## LangSmith

```bash
export LANGSMITH_API_KEY=...
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=symbolic-moe
```

### Trace the baseline router

```bash
python symbolic-moe/route_and_predict.py \
  --input symbolic-moe/validation_pool_skills.jsonl \
  --output symbolic-moe/validation_pool_outputs.jsonl \
  --langsmith-project validation-profile_logodds \
  --langsmith-tags baseline,validation
```

### Trace the Bernoulli router

```bash
python symbolic-moe/route_and_predict_nb.py \
  --input symbolic-moe/validation_pool_skills.jsonl \
  --output symbolic-moe/validation_pool_outputs_skill_nb_top2.jsonl \
  --top-k 2 \
  --langsmith-project validation-skill_nb \
  --langsmith-tags bernoulli,validation
```

### Upload LangSmith datasets

```bash
python symbolic-moe/langsmith_workflows.py upload-dataset \
  --input symbolic-moe/validation_pool.jsonl \
  --dataset-name symbolic-moe-validation

python symbolic-moe/langsmith_workflows.py upload-dataset \
  --input symbolic-moe/test_pool.jsonl \
  --dataset-name symbolic-moe-test
```

### Create a review queue from traced runs

```bash
python symbolic-moe/langsmith_workflows.py queue-review \
  --projects validation-profile_logodds,validation-skill_nb \
  --queue-name symbolic-moe-hard-cases \
  --low-margin-threshold 0.25 \
  --include-cross-task-overlap
```

### Backfill traces from existing routed outputs

```bash
python symbolic-moe/langsmith_workflows.py backfill-traces \
  --input symbolic-moe/validation_pool_outputs.jsonl \
  --router-name profile_logodds \
  --router-script route_and_predict.py \
  --dataset-name symbolic-moe-validation \
  --langsmith-project validation-profile_logodds \
  --langsmith-tags baseline,validation

python symbolic-moe/langsmith_workflows.py backfill-traces \
  --input symbolic-moe/validation_pool_outputs_skill_nb_top2.jsonl \
  --router-name skill_nb \
  --router-script route_and_predict_nb.py \
  --selection-policy top_k_2 \
  --dataset-name symbolic-moe-validation \
  --langsmith-project validation-skill_nb \
  --langsmith-tags bernoulli,validation
```

### Summarize hard cases locally

```bash
python symbolic-moe/langsmith_workflows.py summarize-hard-cases \
  --baseline-input symbolic-moe/validation_pool_outputs.jsonl \
  --compare-input symbolic-moe/validation_pool_outputs_skill_nb_top2.jsonl \
  --out symbolic-moe/tables/langsmith_validation_hard_case_summary.txt

python symbolic-moe/langsmith_workflows.py summarize-hard-cases \
  --baseline-input symbolic-moe/test_pool_outputs.jsonl \
  --compare-input symbolic-moe/test_pool_outputs_skill_nb_top2.jsonl \
  --out symbolic-moe/tables/langsmith_test_hard_case_summary.txt
```
