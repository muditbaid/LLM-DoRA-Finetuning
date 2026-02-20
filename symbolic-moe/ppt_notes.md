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
- Most frequent skills: general_insult (1526), dehumanization (1322), coded_hostility (1250)
- Mid-frequency: stereotype_invocation (682), identity_targeting (611), threatening_language (608)
- Rare skills: identity_based_bullying (329), non_targeted_profanity (128), appearance_based_bullying (69), sexual_harassment (27), age_based_bullying (19)

Per-label skill coverage (top skills per label):
- hate: dehumanization (291), stereotype_invocation (201), general_insult (177), coded_hostility (151), identity_targeting (149)
- not hate: general_insult (126), dehumanization (116), coded_hostility (91), stereotype_invocation (68), identity_targeting (63)
- threat: threatening_language (341), coded_hostility (242), general_insult (223), dehumanization (137), identity_targeting (77)
- not threat: general_insult (38), dehumanization (29), stereotype_invocation (27), coded_hostility (25), identity_targeting (14)
- bully: general_insult (654), dehumanization (514), coded_hostility (512), stereotype_invocation (261), identity_targeting (204)
- not_bully: general_insult (16), dehumanization (12), coded_hostility (9), threatening_language (5), identity_targeting (3)
- offensive: general_insult (234), coded_hostility (158), dehumanization (154), identity_targeting (75), stereotype_invocation (71)
- not offensive: dehumanization (69), coded_hostility (62), general_insult (58), stereotype_invocation (28), identity_targeting (26)

Empty-skill rate by label (profile pool):
- hate: 1522/1923 (79.1%), not hate: 1330/1577 (84.3%)
- threat: 1231/1750 (70.3%), not threat: 1669/1750 (95.4%)
- bully: 1899/2807 (67.7%), not_bully: 667/693 (96.2%)
- offensive: 1442/1787 (80.7%), not offensive: 1584/1713 (92.5%)

## 3) Routing System (high-level)

- Skill inference tags each post with a small set of skills from `skills.txt`.
- Routing weight per expert uses **per-skill log-odds + expert prior**:
  - per-skill log-odds: `logit((correct + 1) / (total + 2))` from profile stats
  - expert prior: `logit((total_correct + 1) / (total_seen + 2))`
  - final weight = prior + sum(per-skill log-odds for the sample’s skills
- Experts are selected if their weight is within `ALPHA * max_weight` for the post.
- Selected experts run and return predictions; outputs are aggregated for evaluation.

## 4) Routing + Prediction Results (validation)

Validation set (3672 samples; 918 per dataset):
- Overall accuracy: **0.8979** (3297/3672)
- Per-dataset accuracy:
  - dynahate: **0.9412** (864/918)
  - jigsaw_threat: **0.9673** (888/918)
  - kaggle_cyberbullying: **0.9401** (863/918)
  - tweeteval_offensive: **0.7429** (682/918)
