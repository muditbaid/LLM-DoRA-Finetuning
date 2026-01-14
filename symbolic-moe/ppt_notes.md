# Symbolic-MoE PPT Notes

## 1) Skill Set (skills.txt)

```
identity_targeting
 dehumanization
 stereotype_invocation
 coded_hostility
 non_targeted_profanity
 general_insult
 threatening_language
 directed_insult
 identity_based_bullying
 appearance_based_bullying
 age_based_bullying
 sexual_harassment
```

Why these skills:
- They are **exclusive variants** designed to separate experts (e.g., hate vs offense vs threat vs bullying).
- They provide **interpretable signals** for routing, making decisions explainable in the pipeline.

## 2) Profiles.json Contents + Skill Stats

What profiles.json contains (per expert):
- `stats[skill]`: `{correct, total}` counts for each skill from the profile pool
- `skill_scores`: normalized per-skill correctness (derived from `stats`)
- `total_seen`, `total_correct`, `accuracy`: overall expert reliability on profile pool

Skill coverage on the profile pool (14,000 rows total):
- Most frequent skills: directed_insult (6028), stereotype_invocation (5865), identity_targeting (3220)
- Mid-frequency: coded_hostility (2984), dehumanization (2983), general_insult (2138), non_targeted_profanity (1250)
- Rare skills: identity_based_bullying (937), appearance_based_bullying (280), age_based_bullying (265), sexual_harassment (66)
- Note: threatening_language appears rarely or not at all in the current profile pool

Per-label skill coverage (top skills per label):
- hate: stereotype_invocation (1352), directed_insult (804), dehumanization (788), identity_targeting (540)
- not hate: stereotype_invocation (650), directed_insult (500), dehumanization (406), general_insult (318)
- threat: directed_insult (754), identity_targeting (367), dehumanization (363), coded_hostility (358)
- not threat: stereotype_invocation (645), directed_insult (483), dehumanization (321), identity_targeting (313)
- bully: directed_insult (1659), stereotype_invocation (1497), identity_targeting (700), coded_hostility (586), identity_based_bullying (473)
- not_bully: directed_insult (165), identity_targeting (115), coded_hostility (99), stereotype_invocation (88)
- offensive: directed_insult (1092), stereotype_invocation (790), coded_hostility (676), identity_targeting (405)
- not offensive: stereotype_invocation (605), directed_insult (571), identity_targeting (483), coded_hostility (396)

Empty-skill rate by label (profile pool):
- hate: 107/1923 (5.6%), not hate: 206/1577 (13.1%)
- threat: 278/1750 (15.9%), not threat: 447/1750 (25.5%)
- bully: 212/2807 (7.6%), not_bully: 259/693 (37.4%)
- offensive: 50/1787 (2.8%), not offensive: 359/1713 (21.0%)

## 3) Routing System (high-level)

- Skill inference tags each post with a small set of skills from `skills.txt`.
- Routing weight per expert uses **per-skill log-odds + expert prior**:
  - per-skill log-odds: `logit((correct + 1) / (total + 2))` from profile stats
  - expert prior: `logit((total_correct + 1) / (total_seen + 2))`
  - final weight = prior + sum(per-skill log-odds for the sample’s skills
- Experts are selected if their weight is within `ALPHA * max_weight` for the post.
- Selected experts run and return predictions; outputs are aggregated for evaluation.

## 4) Routing + Prediction Results (validation)

Validation set (784 samples; 196 per dataset):
- Overall accuracy: **0.9082** (712/784)
- Per-dataset accuracy:
  - dynahate: **0.9337** (183/196)
  - jigsaw_threat: **0.9337** (183/196)
  - kaggle_cyberbullying: **0.9541** (187/196)
  - tweeteval_offensive: **0.8112** (159/196)
