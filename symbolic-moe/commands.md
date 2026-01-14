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
nohup python symbolic-moe/skill_inference.py \
  --input symbolic-moe/profile_pool.jsonl \
  --output symbolic-moe/profile_pool_skills.jsonl \
  --runs 3 --min-count 2 --max-new-tokens 64 \
  > profile_skill_inference.log 2>&1 &

nohup python symbolic-moe/skill_inference.py \
  --input symbolic-moe/validation_pool.jsonl \
  --output symbolic-moe/validation_pool_skills.jsonl \
  --runs 3 --min-count 2 --max-new-tokens 64 \
  > val_skill_inference.log 2>&1 &

nohup python symbolic-moe/skill_inference.py \
  --input symbolic-moe/test_pool.jsonl \
  --output symbolic-moe/test_pool_skills.jsonl \
  --runs 3 --min-count 2 --max-new-tokens 64 \
  > test_skill_inference.log 2>&1 &
```

### 2) Build profiles from profile pool skills

```bash
nohup python symbolic-moe/build_profiles.py \
  --input symbolic-moe/profile_pool_skills.jsonl \
  > build_profiles.log 2>&1 &
```

### 3) Route + predict (validation + test)

```bash
nohup python symbolic-moe/route_and_predict.py \
  --input symbolic-moe/validation_pool_skills.jsonl \
  --output symbolic-moe/validation_pool_outputs.jsonl \
  > route_and_predict.log 2>&1 &

nohup python symbolic-moe/route_and_predict.py \
  --input symbolic-moe/test_pool_skills.jsonl \
  --output symbolic-moe/test_pool_outputs.jsonl \
  > route_and_predict_test.log 2>&1 &
```
