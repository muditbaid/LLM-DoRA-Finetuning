# Symbolic-MoE Commands

## 1. Infer skills for the core pools

```bash
python symbolic-moe/skill_inference.py \
  --input symbolic-moe/profile_pool.jsonl \
  --output symbolic-moe/profile_pool_skills.jsonl \
  --batch-size 32 \
  --runs 5 \
  --min-count 2 \
  --max-new-tokens 64 \
  --max-input-tokens 1536 \
  --quantization 8bit

python symbolic-moe/skill_inference.py \
  --input symbolic-moe/validation_pool.jsonl \
  --output symbolic-moe/validation_pool_skills.jsonl

python symbolic-moe/skill_inference.py \
  --input symbolic-moe/test_pool.jsonl \
  --output symbolic-moe/test_pool_skills.jsonl
```

## 2. Build expert profiles

```bash
python symbolic-moe/build_profiles.py \
  --input symbolic-moe/profile_pool_skills.jsonl
```

## 3. Route and predict with the NB top-2 router

```bash
python symbolic-moe/route_and_predict_nb.py \
  --input symbolic-moe/validation_pool_skills.jsonl \
  --output symbolic-moe/validation_outputs_nb_top2.jsonl \
  --top-k 2

python symbolic-moe/route_and_predict_nb.py \
  --input symbolic-moe/test_pool_skills.jsonl \
  --output symbolic-moe/test_outputs_nb_top2.jsonl \
  --top-k 2
```

## 4. Evaluate routed outputs

```bash
python symbolic-moe/evaluate_outputs.py \
  --input symbolic-moe/validation_outputs_nb_top2.jsonl \
  --metrics-out symbolic-moe/validation_metrics_nb_top2.txt

python symbolic-moe/evaluate_outputs.py \
  --input symbolic-moe/test_outputs_nb_top2.jsonl \
  --metrics-out symbolic-moe/test_metrics_nb_top2.txt
```

## 5. Single-post prediction

```bash
python symbolic-moe/predict_post.py \
  --post "I will find you and hurt you."
```

## LangSmith

```bash
export LANGSMITH_API_KEY=...
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=symbolic-moe
```

### Trace skill inference

```bash
python symbolic-moe/skill_inference.py \
  --input symbolic-moe/validation_pool.jsonl \
  --output symbolic-moe/validation_pool_skills.jsonl \
  --langsmith-project symbolic-moe-skill-inference \
  --langsmith-tags validation,skills
```

### Trace the NB top-2 router

```bash
python symbolic-moe/route_and_predict_nb.py \
  --input symbolic-moe/validation_pool_skills.jsonl \
  --output symbolic-moe/validation_outputs_nb_top2.jsonl \
  --top-k 2 \
  --langsmith-project symbolic-moe-router-nb-top2 \
  --langsmith-tags validation,bernoulli
```

### Trace evaluation

```bash
python symbolic-moe/evaluate_outputs.py \
  --input symbolic-moe/validation_outputs_nb_top2.jsonl \
  --metrics-out symbolic-moe/validation_metrics_nb_top2.txt \
  --langsmith-project symbolic-moe-eval \
  --langsmith-tags validation,evaluation
```

### Upload datasets for reference examples

```bash
python symbolic-moe/langsmith_workflows.py upload-dataset \
  --input symbolic-moe/validation_pool.jsonl \
  --dataset-name symbolic-moe-validation

python symbolic-moe/langsmith_workflows.py upload-dataset \
  --input symbolic-moe/test_pool.jsonl \
  --dataset-name symbolic-moe-test
```

### Backfill traces from routed outputs

```bash
python symbolic-moe/langsmith_workflows.py backfill-traces \
  --input symbolic-moe/validation_outputs_nb_top2.jsonl \
  --router-name skill_nb \
  --router-script route_and_predict_nb.py \
  --selection-policy top_k_2 \
  --dataset-name symbolic-moe-validation \
  --langsmith-project symbolic-moe-router-nb-top2 \
  --langsmith-tags validation,bernoulli
```

### Create a review queue for hard cases

```bash
python symbolic-moe/langsmith_workflows.py queue-review \
  --projects symbolic-moe-router-nb-top2,symbolic-moe-eval \
  --queue-name symbolic-moe-hard-cases \
  --low-margin-threshold 0.25 \
  --include-cross-task-overlap
```
