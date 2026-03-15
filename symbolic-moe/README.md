# Symbolic-MoE Branch Notes (Current Working State)

This README summarizes the debugging and ablation work completed in this branch for symbolic routing quality, with focus on skill inference reliability and profile construction correctness.

## Problem Statement

Routing precision was weak because:

- many posts had empty inferred skills (`Skills: []`),
- routing fell back to priors too often,
- threat/bully experts were selected too broadly for non-threat/non-bully posts.

The branch objective was to reduce no-skill inference behavior and remove profile contamination so expert profiles reflect inferred skills only.

## What Was Fixed

### 1) Skill parsing contamination (prompt echo)

`symbolic-moe/skill_parsing.py` was corrected so parsing is strict:

- parse only explicit `Skills: [...]` lines,
- if multiple `Skills:` lines exist, use the **last** one,
- removed fallback behavior that scanned full raw text for skill tokens.

This prevents prompt/instruction echo from becoming false positive skills.

### 2) Build-profiles leakage and scoring semantics

`symbolic-moe/build_profiles.py` was fixed in two places:

- removed fallback that injected `rec["label"]` as a pseudo-skill when inferred skills were empty,
- excluded `"none"` from profile scoring accumulation (`stats`/`skill_scores_raw`).

Result: profiles are now based on ontology skills only, without gold-label leakage and without `none` affecting skill scores.

### 3) Main skill-inference defaults and behavior

`symbolic-moe/skill_inference.py` defaults and behavior now match branch runtime:

- quantization default: `8bit`
- generation: `temperature=0.2`, `top_p=0.95`, `do_sample=True`
- voting: `runs=5`, `min_count=2`
- left padding enabled for decoder-only generation (`tokenizer.padding_side='left'`)
- batching default: `--batch-size 32`
- no backfill path in the main script
- prompt updated to strict one-line schema output with explicit anti-echo instructions
- prompt rules include: select `0-5` skills, evidence-ranked, no weak guess filling

## Data Fix: Jigsaw Negative Resampling

For profile/validation pool construction, jigsaw negatives were resampled using strict true negatives (all target label columns `== 0.0`) from the unintended-bias source.

Updated pools:

- `symbolic-moe/profile_pool.jsonl` (replaced jigsaw `not threat` rows)
- `symbolic-moe/validation_pool.jsonl` (replaced jigsaw `not threat` rows)

Backups:

- `symbolic-moe/profile_pool.jsonl.bak_true_neg_swap`
- `symbolic-moe/validation_pool.jsonl.bak_true_neg_swap`
- `symbolic-moe/profile_pool.jsonl.bak_strict31_resample`

## Current Profile-Pool Skill Stats (latest run)

From `symbolic-moe/profile_pool_skills.jsonl` (`14000` rows):

- `empty_rate_all = 0.310071`
- `empty_rate_positive = 0.127858`
- `empty_rate_negative = 0.572824`
- `avg_skills_per_post = 1.830786`
- `avg_skills_positive = 2.455425`
- `avg_skills_negative = 0.930054`
- `all11_rate = 0.0`
- `backfill_used_count = 0`

Per-dataset avg skills/post:

- `dynahate = 2.116286`
- `jigsaw_threat = 1.388286`
- `kaggle_cyberbullying = 2.391714`
- `tweeteval_offensive = 1.426857`

## Skill-Dataset Diagnostics (latest)

### `P(skill|dataset)` highlights

- `threatening_language` strongest in `jigsaw_threat` (`0.288571`)
- `stereotype_invocation` strongest in `dynahate` (`0.313714`)
- `identity_based_bullying` strongest in `kaggle_cyberbullying` (`0.071143`)
- `general_insult` strongest in `kaggle_cyberbullying` (`0.415714`)

### `P(dataset|skill)` highlights

- `threatening_language`: jigsaw-dominant (`0.602985`)
- `stereotype_invocation`: dynahate-dominant (`0.505991`)
- `identity_based_bullying`: kaggle-dominant (`0.656992`)

### Specialization and entropy

- highest specialization (`max P(skill|d) - second max`): `threatening_language = 0.213143`
- lowest specialization: `age_based_bullying = 0.003429`
- lowest entropy `H(dataset|skill)`: `identity_based_bullying = 1.321858`
- highest entropy: `coded_hostility = 1.989437`

### Dataset separability from skills

- multiclass logistic regression using skills only:
  - accuracy: `0.454286`
  - random baseline: `0.25`

Interpretation: skills carry useful but moderate dataset-separation signal; routing should improve versus pre-fix contamination but remains partially overlapping by design.

## Ablation Notes (120-post controlled pool)

Ablation work in `symbolic-moe/ablation-skill-inference/` confirmed:

- stricter parse + anti-echo prompt removed all-11 artifacts,
- voting (`runs=5`, `min_count=2`) remained stable,
- stricter evidence rules reduced over-assignment (lower avg skills/post) while preserving low format error.

## Recommended Regeneration Order

After any pool or inference config change, rerun in this order:

1. `skill_inference.py` on `profile_pool.jsonl` and `validation_pool.jsonl`
2. `build_profiles.py`
3. routing/evaluation metrics

Without reruns, `*_skills.jsonl` and `profiles.json` can reflect stale sampling/config state.
