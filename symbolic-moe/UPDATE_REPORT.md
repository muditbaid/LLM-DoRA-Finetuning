# Moderation MoE Update (Markdown Deck)

## Slide 1 — Context
- Goal: Symbolic-MoE routing over Llama 3.1 8B + QLoRA experts for hate/offense/bullying/threat.
- Inputs: skills → profiles → routed expert runs; no aggregation.
- Outputs: structured expert predictions per post (`test_pool_outputs.jsonl`).

## Slide 2 — Experts & Finetune Metrics
- dynahate (binary hate): eval_loss 0.0667 @ epoch 3; profile acc 0.94.
- tweeteval_offense (binary offense): eval_loss 0.1415 @ epoch 3; profile acc 0.86.
- kaggle_bully (bully + subtype): accuracy 0.9511, macro F1 0.9509; per-type F1s — age 0.9880, gender 0.9267, ethnicity 0.9881, religion 0.9590, none 0.8925; profile acc 0.96.
- jigsaw_threat (binary threat): eval_loss 0.0263 @ epoch 3; profile acc 0.98.

## Slide 3 — Skill Set
- Skills from `skills.txt`: identity targeting, implicit hate, dehumanization, stereotype, profanity tone, threatening, appearance/age/gender/ethnicity/sexual bullying.
- Used both for profiling (validation pool) and routing (inference).

## Slide 4 — Inference Module
- `model_utils.ExpertModel`: loads base Llama‑3.1‑8B‑Instruct + LoRA adapter per expert.
- Builds prompts from record (system/instruction/input), short deterministic generation (max tokens per expert config).
- Normalizes labels via `build_profiles.normalize_label` (special casing bully outputs).

## Slide 5 — Profiles (Routing Signals)
- `build_profiles.py` on validation pool: for each expert and skill, +1/-1 depending on correctness; stores `skill_scores`, per-skill stats, total_seen/correct, global accuracy.
- Stored in `profiles.json` (top skills per expert: dynahate→dehumanization/stereotype/implicit hate; offense→profanity tone/implicit hate; bully→implicit hate/stereotype; threat→threatening/dehumanization).

## Slide 6 — Routing Logic (`route_and_predict.py`)
- Collect skills from `predicted_skills` (or `skill_tag` fallback) and filter by `SKILL_VOCAB`.
- If no skills: emit `predictions: []`.
- For each expert: weight = sum(skill_scores for mapped skills) × expert accuracy; keep positives, sort, top‑3; fallback to expert matching first skill label if none positive.
- Batch per-expert, run predictions, append `{expert, weight, raw_prediction, normalized_prediction}`.

## Slide 7 — Sample Evaluation (test_pool_outputs)
- Rule: correct if gold `output` appears in any normalized_prediction; for negated gold (not_*), also correct when the positive label is absent.
- Results over 600 test rows: overall 89.5% (537/600).
- Per-dataset: dynahate 88.7% (133/150); kaggle_bully 81.3% (122/150); jigsaw_threat 95.3% (143/150); tweeteval_offensive 92.7% (139/150).
- Script: `symbolic-moe/evaluate_outputs.py` (run with module-aware invocation or add `symbolic-moe` to `PYTHONPATH`).

## Slide 8 — Next Steps
- Tighten offense/bully routing weights or thresholds to lift their per-dataset scores.
- Add a small aggregator/judge if a single consolidated label is desired.
- Expand skills (e.g., harassment intent, self-harm) if coverage gaps appear.
