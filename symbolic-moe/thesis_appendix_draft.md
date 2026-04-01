# Thesis Appendix Draft

This document is a revised appendix plan that removes material already covered in the current dissertation draft PDF. The goal is to keep the appendix genuinely supplementary rather than repeating Chapter 4 and Chapter 5.

## What Is Already In the Dissertation

The current dissertation draft already includes the following in the main body:

- Dataset summary
- Data pool composition
- Expert summary
- Shared training hyperparameters
- Calibration of `\alpha` on the validation pool
- Task-specific benchmark comparison tables for all four experts
- Profile statistics summary
- Validation and test routed-performance comparison tables
- NB top-k calibration summary
- Routing behavior summary
- Representative qualitative cases

These appear in the current PDF as:

- Table 4.1 `Datasets Summary`
- Table 4.2 `Data Pool Composition`
- Table 4.3 `Expert Summary`
- Table 4.4 `Shared Training Hyperparameters`
- Table 4.5 `Calibration of α on validation pool`
- Tables 5.1 to 5.4 benchmark comparisons
- Table 5.5 `Profile Statistics Summary`
- Tables 5.6 to 5.11 routing-performance tables
- Table 5.12 `NB Top-k Calibration candidate summary`
- Table 5.13 `Routing behavior summary`
- Table 5.14 `Representative qualitative cases`

Because these are already in the dissertation, they should not be repeated in the appendix.

## UGA 2025 Constraints That Matter

- The 2025 UGA Graduate School style guide says appendices are optional and should be placed after the bibliography.
- Appendix titles should be formatted like other chapter/section titles.
- The first page of an appendix uses the larger first-page top margin rule, just like a chapter opening page.

## Revised Appendix Structure

After removing duplicated material, the appendix should focus primarily on overflow tables, figures, and diagnostics that are not shown in the dissertation chapters. Since the repository will be public, file-level reproducibility can live on GitHub rather than taking up appendix space.

The leanest useful structure is:

| Appendix | Title | Keep? | Why |
|---|---|---|---|
| A | Full Prompts and Output Contracts | Yes | Useful appendix material that is not fully shown in the main text. |
| B | Label Spaces and Evaluation Rules | Yes | Exact operational rules are better placed in the appendix than in the results narrative. |
| C | Supplementary Configuration Details | Yes, trimmed | Keep only compact settings that support interpretation, not full reproducibility boilerplate. |
| D | Additional Diagnostics Not Already Tabulated in Chapter 5 | Yes | Keep ablations and empty-skill analyses. |
| E | Additional Tables and Figures Not Used in the Main Text | Yes | This should become the main overflow appendix for unused but valuable visuals and tables. |
| F | Extended Qualitative Cases | Usually no | Only keep if you need more examples than the main text already shows. |

If you want the cleanest final thesis, use only Appendices A through E, with Appendix E acting as the main home for overflow figures and tables.

## Appendix A. Full Prompts and Output Contracts

This appendix should keep prompt text verbatim.

### A.1 Expert task prompts

Use the canonical prompts from the dataset artifacts:

| Task | System prompt | Instruction prompt | Source |
|---|---|---|---|
| DynaHate | `Strictly respond only with the label: 'hate' or 'not hate'.` | `You are a helpful Assistant. Your task is to classify the social media post as hate or not hate. Post:` | `data/dynahate_train.jsonl` |
| TweetEval Offensive | `Strictly respond only with the label: 'offensive' or 'not offensive'.` | `You are a helpful Assistant. Your task is to classify the social media post as offensive or not offensive. Post:` | `data/tweeteval_offensive_train.jsonl` |
| Jigsaw Threat | `Strictly respond only with the label: 'threat' or 'not threat'.` | `You are a helpful Assistant. Your task is to classify the social media post as threat or not threat. Post:` | `data/jigsaw_threat_train.jsonl` |
| Kaggle Cyberbullying | `You review social media posts for bullying. Reply with a single line in the format: label: bully|not_bully; type: age|gender|ethnicity|religion|none. Use type: none whenever the post is not bullying.` | Empty instruction in the current JSONL export. | `data/kaggle_cyberbullying_train.jsonl`, `scripts/generate_kaggle_cyberbullying_jsonl.py` |

### A.2 Current symbolic skill-inference prompt

Use the exact current prompt contract from:

- `symbolic-moe/skill_inference.py`
  - `SYSTEM_PROMPT`
  - `USER_TEMPLATE`
- `symbolic-moe/skills.txt`

This section should explicitly state:

- the allowed output contract is `Skills: [...]` or `Skills: []`,
- only controlled-vocabulary skills are allowed,
- the prompt asks for 0 to 5 skills,
- but repeated-sampling vote aggregation can yield more than 5 retained skills in committed outputs.

That implementation detail is important and is not something the main dissertation needs to dwell on.

### A.3 Pilot batch prompts

If you discuss the pilot OpenAI batch experiments in the thesis, include:

- `symbolic-moe/batch_create.py`
  - `SKILLS_PROMPT_TEMPLATE`
  - `LABELS_PROMPT_TEMPLATE`

If the pilot experiments are not mentioned in the final thesis text, omit this subsection entirely.

## Appendix B. Label Spaces, Normalization Rules, and Evaluation Semantics

This appendix should contain exact operational definitions, not high-level prose.

### B.1 Expert label spaces

Use `symbolic-moe/config.py`:

| Expert | Label texts | Normalizer |
|---|---|---|
| `dynahate_hate` | `hate`, `not hate` | `simple` |
| `tweeteval_offense` | `offensive`, `not offensive` | `simple` |
| `jigsaw_threat` | `threat`, `not threat` | `simple` |
| `kaggle_bully` | subtype-bearing bully strings plus `label: not_bully; type: none` | `bully` |

### B.2 Normalization rules

Use:

- `symbolic-moe/build_profiles.py`
- `symbolic-moe/evaluate_outputs.py`
- `scripts/generate_kaggle_cyberbullying_jsonl.py`

Document exactly:

- `simple` normalization
- `bully` normalization
- subtype handling for cyberbullying
- negative-label handling

### B.3 Evaluator correctness rules

This should be written explicitly because it materially affects interpretation of results.

Current logic from `symbolic-moe/evaluate_outputs.py`:

1. Normalize the gold label using the dataset-specific normalizer.
2. Mark an example correct if any routed expert prediction matches the normalized gold label.
3. For negative labels, also mark an example correct if the positive counterpart is absent from all predictions.

This belongs in the appendix even if Chapter 4 or Chapter 5 briefly summarizes evaluation.

### B.4 Pilot normalization note

Only keep this if the pilot batch experiments remain part of the final thesis.

If kept, document:

- `symbolic-moe/batch_fetch.py`
- the `RAW_TO_CANON` mapping

If the pilot path is not discussed in the thesis, remove this subsection.

## Appendix C. Supplementary Configuration Details

This appendix should only keep compact configuration details that help interpretation. Since the repository will be public, the appendix does not need to carry a full reproducibility inventory.

### Keep

- Dataset-specific training settings not already shown in Table 4.4
- Skill-inference runtime defaults and vote settings
- Software versions
- Any hardware note, but only if you can verify it

### Remove

- Dataset summary
- Data pool composition
- Expert summary
- Shared training hyperparameter table

Those are already in Chapter 4.

### C.1 Dataset-specific training settings

The main text already has shared training hyperparameters. The appendix can keep the per-dataset deltas:

| Expert / dataset | Train batch size | Grad accum | Effective batch | Learning rate | Warmup ratio | Max length |
|---|---:|---:|---:|---:|---:|---:|
| DynaHate | 3 | 8 | 24 | 2e-5 | 0.10 | 1024 |
| TweetEval Offensive | 4 | 8 | 32 | 2e-5 | 0.05 | 1024 |
| Jigsaw Threat | 4 | 8 | 32 | 2e-5 | 0.05 | 1024 |
| Kaggle Cyberbullying | 1 | 8 | 8 | 5e-5 | 0.05 | 2048 |

Sources:

- `examples/train_qlora/llama31_dynahate_qlora_sft.yaml`
- `examples/train_qlora/llama31_tweeteval_offensive_qlora_sft.yaml`
- `examples/train_qlora/llama31_jigsaw_qlora_sft.yaml`
- `examples/train_qlora/llama31_kaggle_cyberbullying_qlora_sft.yaml`

### C.2 Skill-inference runtime settings

Use:

- `symbolic-moe/skill_inference.py`
- `symbolic-moe/README.md`

Keep:

- runs
- min-count
- max-new-tokens
- batch size
- max-input-tokens
- generation settings
- note about configuration drift between older branch notes and current mainline script defaults

This is appendix-appropriate because it is too operational for the methodology chapter.

### C.3 Optional software note

Use:

- `saves/llama31-8b/*/qlora/README.md`

Keep only a compact note listing:

- `transformers`
- `peft`
- `bitsandbytes`
- `datasets`
- `tokenizers`
- PyTorch

### C.4 Hardware note

Do not guess the GPU type. If it is not recoverable from committed artifacts, say so plainly.

## Appendix D. Additional Diagnostics Not Already in Chapter 5

This appendix should be selective. It should not repeat the main routing comparison tables, benchmark tables, or representative-case table because those already appear in Chapter 5.

### Keep

- Skill-inference ablations
- Empty-skill diagnostics
- Any low-level diagnostic plots not already included in the dissertation

### Remove

- Benchmark comparison tables
- Profile statistics summary
- Validation and test routing comparison tables
- NB top-k calibration summary
- Routing behavior summary
- Representative qualitative cases

Those are already in Chapter 5.

### D.1 Skill-inference ablations

These ablations are suitable appendix material because they justify the final skill-inference settings without overloading the main results chapter. The committed ablation artifacts evaluate a controlled 120-post pool with the bucket strategy `40 positive_empty_current + 40 positive_non_empty_current + 40 negative`. The relevant sources are:

- `symbolic-moe/ablation-skill-inference/outputs/exp1a_greedy_metrics.txt`
- `symbolic-moe/ablation-skill-inference/outputs/exp1b_temp02_topp095_metrics.txt`
- `symbolic-moe/ablation-skill-inference/outputs/exp2a_runs5_min2_metrics.txt`
- `symbolic-moe/ablation-skill-inference/outputs/exp2b_temperature_sweep_metrics.txt`

#### D.1.1 Decoding and Voting Ablation

| Setting | Sampling | Temperature / Top-p | Runs | Min count | Empty rate (all) | Empty rate (positive) | Empty rate (negative) | Avg. skills/post | Avg. skills/positive post | Format error rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP-1A Greedy decoding | No | disabled | 1 | not used | not reported | 0.0875 | not reported | 2.6417 | not reported | not reported |
| EXP-1B Sampled decoding, no voting | Yes | 0.2 / 0.95 | 1 | not used | 0.1750 | 0.0875 | 0.3500 | 2.6083 | 3.1375 | 0.0000 |
| EXP-2A Sampled decoding with voting | Yes | 0.2 / 0.95 | 5 | 2 | 0.1750 | 0.0875 | 0.3500 | 2.8750 | 3.4125 | 0.0000 |

This ablation supports two practical conclusions. First, switching from greedy decoding to low-temperature sampling did not improve empty-skill rates on positive posts in this controlled pool; both EXP-1A and EXP-1B retained an empty-positive rate of `0.0875`. Second, reintroducing majority voting at the same decoding settings increased the number of retained skills without increasing format errors, raising the average number of skills per post from `2.6083` to `2.8750` and the average number of skills per positive post from `3.1375` to `3.4125`.

#### D.1.2 Temperature Sweep

The temperature sweep keeps `top_p = 0.95`, `runs = 5`, `min_count = 2`, `backfill = none`, and `quantization = 8bit` fixed while varying only the temperature. The results are:

| Temperature | Sampling | Empty rate (all) | Empty rate (positive) | Empty rate (negative) | Avg. skills/post | Avg. skills/positive post | Format error rate |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.0 | No | 0.1750 | 0.0875 | 0.3500 | 2.6417 | 3.2375 | 0.0000 |
| 0.2 | Yes | 0.1750 | 0.0875 | 0.3500 | 2.8333 | 3.4000 | 0.0000 |
| 0.5 | Yes | 0.1667 | 0.0875 | 0.3250 | 2.9667 | 3.5375 | 0.0000 |
| 0.7 | Yes | 0.1583 | 0.0875 | 0.3000 | 3.0667 | 3.6125 | 0.0000 |

Across the tested settings, higher temperature increased skill yield and modestly reduced overall empty-skill rates, especially by reducing empty outputs on negative posts from `0.3500` at temperature `0.0` or `0.2` to `0.3000` at temperature `0.7`. The positive empty-skill rate remained constant at `0.0875` across all four temperature settings in this ablation.

#### D.1.3 Format-Error Note

The retained ablation artifacts report a format-error rate of `0.0000` for EXP-1B, EXP-2A, and every configuration in the temperature sweep. EXP-1A does not report a format-error field in its summary file. The appendix can therefore state that the sampled and vote-based configurations preserved the output contract reliably within the recorded ablation runs.

### D.2 Empty-skill diagnostics

This is one of the strongest appendix sections because empty-skill behavior is central to the routing limitations and is not fully captured by the main dissertation tables. The relevant artifacts are:

- `symbolic-moe/analyze_empty_skills.py`
- `symbolic-moe/charts/empty_skills/profile_pool/*`
- `symbolic-moe/charts/empty_skills/validation_baseline/*`
- `symbolic-moe/charts/empty_skills/validation_nb_top2/*`

The most useful files for direct appendix use are:

- `empty_skill_summary.txt`
- `empty_skill_dataset_summary.csv`
- `empty_skill_examples.txt`

#### D.2.1 Overall Empty-Skill Prevalence

| Artifact | Total rows | Empty-skill rows | Empty-skill rate |
|---|---:|---:|---:|
| Profile pool (`profile_pool_skills.jsonl`) | 14000 | 4341 | 0.3101 |
| Validation pool (`validation_pool_skills.jsonl`) | 3672 | 1190 | 0.3241 |

Empty-skill outputs are therefore not rare edge cases. Roughly one-third of the profiled and validation examples contain no retained skills after inference and vote filtering.

#### D.2.2 Per-Dataset Empty-Skill Rates

Profile-pool empty-skill rates:

| Dataset | Rows | Empty rows | Empty rate |
|---|---:|---:|---:|
| DynaHate | 3500 | 751 | 0.2146 |
| TweetEval Offensive | 3500 | 1292 | 0.3691 |
| SOSNet Cyberbullying | 3500 | 821 | 0.2346 |
| Jigsaw Threat | 3500 | 1477 | 0.4220 |

Validation-pool empty-skill rates:

| Dataset | Rows | Empty rows | Empty rate |
|---|---:|---:|---:|
| DynaHate | 918 | 178 | 0.1939 |
| TweetEval Offensive | 918 | 347 | 0.3780 |
| SOSNet Cyberbullying | 918 | 238 | 0.2593 |
| Jigsaw Threat | 918 | 427 | 0.4651 |

The highest empty-skill rates occur in Jigsaw Threat and TweetEval Offensive, while DynaHate and SOSNet Cyberbullying have lower but still substantial empty-skill fractions.

#### D.2.3 Label Distribution Within Empty-Skill Groups

The empty-skill group is disproportionately negative relative to the nonempty group. In the profile pool, the positive rate is `0.2435` for empty-skill rows versus `0.7465` for nonempty rows. In the validation pool, the positive rate is `0.2345` for empty-skill rows versus `0.7566` for nonempty rows. This means empty-skill behavior is strongly associated with less overtly harmful or less explicitly signaled posts, although it still includes positive examples in every dataset.

At the dataset level, the empty-skill positive rate remains notably lower than the nonempty positive rate. For example, in the profile pool, Jigsaw Threat has an empty positive rate of `0.1361` versus a nonempty positive rate of `0.7657`, while SOSNet Cyberbullying has an empty positive rate of `0.3971` versus a nonempty positive rate of `0.9261`.

#### D.2.4 Text-Feature Differences Between Empty and Nonempty Groups

The group-level summaries suggest that empty-skill posts tend to contain weaker explicit lexical signals. In the profile pool:

- profanity rate is `0.0251` for empty rows versus `0.1370` for nonempty rows,
- average length is `24.02` words for empty rows versus `27.15` for nonempty rows,
- mention rate is slightly higher for empty rows (`0.3550` versus `0.3042`),
- quote rate is lower for empty rows (`0.3227` versus `0.3928`).

In the validation pool:

- profanity rate is `0.0311` for empty rows versus `0.1475` for nonempty rows,
- average length is `30.04` words for empty rows versus `28.73` for nonempty rows,
- mention rate is `0.3370` for empty rows versus `0.3058` for nonempty rows,
- non-ASCII rate is `0.1689` for empty rows versus `0.1285` for nonempty rows.

These summaries support the interpretation that empty-skill failures are tied partly to weak explicit cues, stylistic noise, and posts whose harmfulness is contextual rather than lexically direct.

#### D.2.5 Routing Consequences on the Validation Pool

Because the empty/nonempty split is computed from the same validation skill file, the empty-skill rate is identical across the baseline and NB-top2 analyses. What changes is how routing behaves once the system is forced to operate with or without retained skills.

Baseline validation results for empty-skill rows:

| Dataset | Empty gold-expert recall@1 | Empty gold-expert recall@4 | Empty permissive accuracy |
|---|---:|---:|---:|
| DynaHate | 0.0000 | 1.0000 | 0.9438 |
| TweetEval Offensive | 0.0000 | 1.0000 | 0.8386 |
| SOSNet Cyberbullying | 0.0000 | 1.0000 | 0.9034 |
| Jigsaw Threat | 1.0000 | 1.0000 | 0.9766 |

NB-top2 validation results for empty-skill rows:

| Dataset | Empty gold-expert recall@1 | Empty gold-expert recall@2 | Empty permissive accuracy |
|---|---:|---:|---:|
| DynaHate | 0.0000 | 0.0000 | 0.7697 |
| TweetEval Offensive | 0.0000 | 1.0000 | 0.8386 |
| SOSNet Cyberbullying | 0.0000 | 0.0000 | 0.5756 |
| Jigsaw Threat | 1.0000 | 1.0000 | 0.9766 |

This pattern shows that empty-skill rows are especially damaging when routing becomes sparse. Under baseline routing, broad expert activation preserves high permissive accuracy even when no skills are retained. Under NB-top2 routing, empty-skill rows remain particularly problematic for DynaHate and SOSNet Cyberbullying because the correct expert often drops out of the routed top-2 set.

#### D.2.6 Qualitative Audit Trail

The `empty_skill_examples.txt` files provide an appendix-ready audit trail for manual inspection. They show that empty-skill behavior includes both false negatives on clearly harmful posts and plausible abstentions on subtle or context-dependent non-harmful posts. Because these files contain explicit harmful language, the appendix should include a short content warning if any examples are reproduced verbatim.

### D.3 Optional low-level diagnostics

Only keep these if they are not already shown in the dissertation:

- additional dataset/profile charts from `symbolic-moe/charts/datasets/*`
- additional profile charts from `symbolic-moe/charts/profiles/*`

If the corresponding heatmaps are already in Chapter 5, do not repeat them here.

## Appendix E. Additional Tables and Figures Not Used in the Main Text

This appendix should become the main home for overflow charts and tables that are useful, thesis-relevant, and not already shown in the dissertation body.

### E.1 Tables worth moving into the appendix

The strongest candidates are:

- `symbolic-moe/tables/dataset_specific_training_settings.tex`
  - good if you want one compact table of per-dataset training differences without expanding Chapter 4
- `symbolic-moe/tables/skill_vocabulary.tex`
  - useful if you want the full skill list and definitions in the appendix rather than the main text
- `symbolic-moe/results/per_dataset_routed_performance.tex`
  - useful as an appendix delta table showing where NB helps or hurts relative to baseline
- `symbolic-moe/tables/test_routing_comparison.tex`
  - keep only if it is not already represented by Tables 5.9 to 5.11 in the dissertation

### E.2 Figures worth moving into the appendix

The strongest unused or overflow figures appear to be:

- `symbolic-moe/charts/results/gold_expert_recall_topk.pdf`
  - useful if you want a visual for routing-depth behavior rather than only the tabulated version
- `symbolic-moe/charts/empty_skills/validation_baseline/empty_skill_rate_by_dataset.pdf`
- `symbolic-moe/charts/empty_skills/validation_nb_top2/empty_skill_rate_by_dataset.pdf`
- `symbolic-moe/charts/empty_skills/validation_baseline/empty_skill_permissive_accuracy_by_dataset.pdf`
- `symbolic-moe/charts/empty_skills/validation_nb_top2/empty_skill_permissive_accuracy_by_dataset.pdf`
- `symbolic-moe/charts/empty_skills/profile_pool/empty_skill_feature_rates.pdf`
- `symbolic-moe/charts/empty_skills/validation_baseline/empty_skill_feature_rates.pdf`
- `symbolic-moe/charts/empty_skills/validation_nb_top2/empty_skill_feature_rates.pdf`

These are strong appendix figures because they directly support the diagnostic story but would overload Chapter 5 if all were included there.

### E.3 Profile and distribution visuals to consider

If they are not already shown elsewhere, the following are good appendix candidates:

- per-dataset skill distribution bar charts in `symbolic-moe/charts/datasets/*.pdf`
- per-expert skill distribution charts in `symbolic-moe/charts/profiles/*_skill_distribution.pdf`
- profile-card style summaries such as `symbolic-moe/charts/profiles/dynahate_hate_profile_card.pdf`

These are especially useful if you want the appendix to visually demonstrate expert specialization without adding more tables to the main text.

### E.4 What should stay on GitHub instead

Since the repository will be public, these do not need a dedicated appendix section unless your committee specifically asks for them:

- file-by-file artifact map
- full command inventory
- pipeline reproduction order
- exhaustive list of JSONL outputs

A short sentence in the thesis can simply point readers to the public repository for those materials.

## Appendix F. Extended Qualitative Cases

This appendix should usually be removed because the dissertation already contains representative qualitative cases in Table 5.14.

Only keep Appendix F if:

- you want a larger case archive than Table 5.14,
- and the extra cases are clearly additional rather than repeated.

If kept, it should contain only overflow examples not already discussed in Chapter 5.

## Recommended Final Decision

For the current dissertation draft, the cleanest appendix set is:

1. Appendix A. Full Prompts and Output Contracts
2. Appendix B. Label Spaces, Normalization Rules, and Evaluation Semantics
3. Appendix C. Supplementary Configuration Details
4. Appendix D. Additional Diagnostics Not Already in Chapter 5
5. Appendix E. Additional Tables and Figures Not Used in the Main Text

If the GitHub repository is public, deprioritize the appendix as a reproducibility archive. Use it instead as a place to surface the strongest unused charts, overflow tables, and diagnostic visuals that support the thesis but do not fit cleanly into Chapters 4 and 5.
