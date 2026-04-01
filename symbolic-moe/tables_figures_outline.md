# Tables and Figures Outline

This outline remaps the thesis tables and figures to the current dissertation structure, the current Bernoulli top-k routing pipeline, and the appendix plan in [thesis_appendix_draft.md](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/thesis_appendix_draft.md).

It is grounded in:

- the current dissertation PDF: [dissertation.pdf](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/downloads/dissertation.pdf)
- the current thesis-ready tables in [tables](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/tables)
- the current charts in [charts](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/charts)
- the current metric summaries in [results](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/results)

The current chapter structure relevant to this outline is:

- Chapter 4: Methodology
- Chapter 5: Results and Discussion
- Appendix A-F as defined in the appendix draft

## General Notes

- Keep the main body focused on a compact set of high-signal figures and tables.
- Move dense metric dumps, split-by-split comparisons, and extended diagnostics into the appendix.
- Do not keep any figure or table tied to the old alpha-threshold router as a main result. The current method uses Bernoulli routing with top-k selection and validation-based calibration.
- In main-text results, treat `top-2` as the selected operating point and `top-1/top-3` as comparison settings.

## Main-Text Figures

### Figure 4.1. End-to-End Symbolic-MoE Pipeline

- Placement:
  - Chapter 4, Section 4.1 `Methodological Overview`
- Content:
  - input post
  - symbolic skill inference
  - inferred symbolic skills
  - Bernoulli routing module
  - ranked experts
  - top-k selection
  - selected experts
  - expert predictions
  - post-level output assembly
- Include:
  - shared base model
  - four expert adapters
  - possibility of multiple routed experts
- Status:
  - already present in the dissertation as Figure 4.1

### Figure 4.2. Skill Inference Workflow

- Placement:
  - Chapter 4, Section 4.7 `Skill Inference`
- Content:
  - one post
  - repeated skill inference runs
  - sampled outputs
  - vote counting
  - min-count retention
  - final canonical skill set
- Optional annotations:
  - runs
  - temperature
  - top-p
  - min-count
- Status:
  - already present in the dissertation as Figure 4.2

### Figure 5.1. Dataset Skill Heatmap

- Placement:
  - Chapter 5, Section 5.3 `Skill Distributions Across Datasets`
- Source:
  - [datasets_skill_heatmap.png](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/charts/datasets/datasets_skill_heatmap.png)
- Purpose:
  - main aggregate view of symbolic skill distributions across datasets

### Figures 5.2-5.5. Per-Dataset Skill Distributions

- Placement:
  - Chapter 5, Section 5.3 `Skill Distributions Across Datasets`
- Sources:
  - [dynahate_skill_distribution.png](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/charts/datasets/dynahate_skill_distribution.png)
  - [jigsaw_threat_skill_distribution.png](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/charts/datasets/jigsaw_threat_skill_distribution.png)
  - [kaggle_cyberbullying_skill_distribution.png](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/charts/datasets/kaggle_cyberbullying_skill_distribution.png)
  - [tweeteval_offensive_skill_distribution.png](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/charts/datasets/tweeteval_offensive_skill_distribution.png)
- Purpose:
  - show dominant and overlapping skills for each dataset

### Figure 5.6. Expert Profile Heatmap

- Placement:
  - Chapter 5, Section 5.4 `Expert Profile Characteristics`
- Source:
  - [profiles_skill_heatmap.png](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/charts/profiles/profiles_skill_heatmap.png)
- Purpose:
  - show aggregate skill-conditioned competence patterns across experts

### Figure 5.11. Gold-Expert Recall at Top-k

- Placement:
  - Chapter 5, Section 5.6 `Routing Behavior and Expert Selection`
- Status:
  - not yet rendered as a committed plot; should be created from:
    - [baseline_routing_metrics.md](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/results/baseline_routing_metrics.md)
    - [nb_routing_metrics.md](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/results/nb_routing_metrics.md)
- Content:
  - x-axis: `k = 1, 2, 3, 4`
  - y-axis: gold-expert recall
  - lines:
    - baseline
    - NB top-k validation
    - optionally test NB top-k
- Purpose:
  - visually justify the improvement in expert ranking and the top-k tradeoff

### Figure 5.12. Empty-Skill Rate by Dataset

- Placement:
  - Chapter 5, Section 5.8 `Limitations and Error Analysis`
- Source:
  - [empty_skill_rate_by_dataset.png](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/charts/empty_skills/validation_baseline/empty_skill_rate_by_dataset.png)
- Purpose:
  - show one of the main symbolic-layer failure modes

### Figure 5.13. Empty-Skill Positive Rate by Dataset

- Placement:
  - Chapter 5, Section 5.8 `Limitations and Error Analysis`
- Source:
  - [empty_skill_positive_rate_by_dataset.png](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/charts/empty_skills/validation_baseline/empty_skill_positive_rate_by_dataset.png)
- Purpose:
  - show how empty-skill behavior differs by class balance and dataset

### Figure 5.14. Qualitative Routing Examples

- Placement:
  - Chapter 5, Section 5.7 `Interpretability of Symbolic Routing`
- Status:
  - to be assembled manually from committed output examples
- Suggested source cases:
  - Appendix E candidates in [thesis_appendix_draft.md](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/thesis_appendix_draft.md)
- Purpose:
  - compact visual illustration of post -> skills -> routed experts -> predictions

## Main-Text Tables

### Table 4.1. Dataset Summary

- Placement:
  - Chapter 4, Section 4.3 `Datasets`
- Source:
  - [dataset_summary.tex](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/tables/dataset_summary.tex)
- Purpose:
  - main dataset reference table

### Table 4.2. Data Pool Composition

- Placement:
  - Chapter 4, Section 4.3 `Datasets`
- Source:
  - [data_pool_composition.tex](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/tables/data_pool_composition.tex)
- Purpose:
  - clarify training/profile/validation/test pool roles

### Table 4.3. Expert Model Summary

- Placement:
  - Chapter 4, Section 4.4 `Expert Model Construction`
- Source:
  - [expert_summary.tex](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/tables/expert_summary.tex)
- Purpose:
  - quick reference for the four task-specific experts

### Table 4.4. Shared Training Hyperparameters

- Placement:
  - Chapter 4, Section 4.5 `Fine-Tuning Procedure`
- Source:
  - [shared_training_hyperparameters.tex](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/tables/shared_training_hyperparameters.tex)

### Table 4.5. Dataset-Specific Training Settings

- Placement:
  - Chapter 4, Section 4.5 `Fine-Tuning Procedure`
- Source:
  - [dataset_specific_training_settings.tex](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/tables/dataset_specific_training_settings.tex)

### Table 4.6. Skill Vocabulary

- Placement:
  - Chapter 4, Section 4.7 `Skill Inference`
- Source:
  - [skill_vocabulary.tex](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/tables/skill_vocabulary.tex)

### Table 4.7. Profile Statistics Summary

- Placement:
  - Chapter 4, Section 4.6 `Expert Profiling`
  - or Chapter 5, Section 5.4 if you want the stronger analytic emphasis there
- Source:
  - [profile_statistics_summary.tex](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/tables/profile_statistics_summary.tex)
- Recommendation:
  - use a compact version in Chapter 4 only if the methodology chapter needs it; otherwise reserve it for Chapter 5

### Table 5.1. Expert and Benchmark Task Summary

- Placement:
  - Chapter 5, Section 5.2 `Task-Specific Expert Performance`
- Source:
  - likely derived from [expert_summary.tex](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/tables/expert_summary.tex)
- Purpose:
  - orient the reader before the benchmark-comparison tables

### Tables 5.2-5.5. Benchmark Comparison Tables

- Placement:
  - Chapter 5, Section 5.2 `Task-Specific Expert Performance`
- Sources:
  - [dynahate_benchmark_comparison.tex](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/tables/dynahate_benchmark_comparison.tex)
  - [tweeteval_offensive_benchmark_comparison.tex](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/tables/tweeteval_offensive_benchmark_comparison.tex)
  - [sosnet_benchmark_comparison.tex](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/tables/sosnet_benchmark_comparison.tex)
  - [jigsaw_threat_benchmark_comparison.tex](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/tables/jigsaw_threat_benchmark_comparison.tex)

### Table 5.6. Profile Statistics Summary

- Placement:
  - Chapter 5, Section 5.4 `Expert Profile Characteristics`
- Source:
  - [profile_statistics_summary.tex](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/tables/profile_statistics_summary.tex)

### Table 5.7. Main Validation Routing Results

- Placement:
  - Chapter 5, Section 5.5 `End-to-End Routed System Performance`
- Source:
  - [validation_routing_comparison.tex](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/tables/validation_routing_comparison.tex)
- Recommendation:
  - keep this compact and centered on baseline vs NB `top-1/top-2/top-3`

### Table 5.8. Main Test Routing Results

- Placement:
  - Chapter 5, Section 5.5 `End-to-End Routed System Performance`
- Status:
  - not yet rendered as a LaTeX table; derive from:
    - [baseline_all_metrics.md](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/results/baseline_all_metrics.md)
    - [nb_all_metrics.md](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/results/nb_all_metrics.md)
- Purpose:
  - test-side counterpart to the validation comparison table

### Table 5.9. Top-k Calibration Summary

- Placement:
  - Chapter 5, Section 5.6 `Routing Behavior and Expert Selection`
- Source:
  - [nb_topk_calibration.md](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/results/nb_topk_calibration.md)
- Purpose:
  - document why `k=2` was selected

### Table 5.10. Per-Dataset Routed Performance

- Placement:
  - Chapter 5, Section 5.5 or 5.6
- Status:
  - derive from:
    - [baseline_all_metrics.md](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/results/baseline_all_metrics.md)
    - [nb_all_metrics.md](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/results/nb_all_metrics.md)
- Purpose:
  - show where NB helps and where it degrades

### Table 5.11. Routing Behavior Summary

- Placement:
  - Chapter 5, Section 5.6 `Routing Behavior and Expert Selection`
- Source:
  - derive from [baseline_routing_metrics.md](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/results/baseline_routing_metrics.md) and [nb_routing_metrics.md](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/results/nb_routing_metrics.md)
- Recommended columns:
  - setting
  - routing top-1
  - gold-expert recall@1
  - gold-expert recall@2
  - gold-expert recall@3
  - selected `k`

### Table 5.12. Representative Error/Case Table

- Placement:
  - Chapter 5, Section 5.7 or 5.8
- Status:
  - manual assembly from committed output examples
- Purpose:
  - compact examples of successful routing, routing drift, and empty-skill failure

## Appendix Figures and Tables

These should not crowd the main body, but they are valuable to keep in the appendix because they support claims already made in the thesis.

### Appendix A

- expert prompt tables
- current symbolic skill-inference prompt

### Appendix B

- expert label-space table
- normalization rules table if needed
- evaluator correctness-rule box or numbered list
- full skill vocabulary table if not placed in Chapter 4

### Appendix C

- shared training hyperparameters
- dataset-specific training settings
- software versions
- runtime settings

### Appendix D: Extended Results, Ablations, and Diagnostics

Keep the dense result artifacts here:

- full baseline summaries:
  - [baseline_all_metrics.md](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/results/baseline_all_metrics.md)
  - [baseline_routing_metrics.md](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/results/baseline_routing_metrics.md)
- full NB summaries:
  - [nb_all_metrics.md](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/results/nb_all_metrics.md)
  - [nb_routing_metrics.md](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/results/nb_routing_metrics.md)
- full raw metric dumps:
  - validation and test metric `.txt` files
- skill-inference ablation tables:
  - from `symbolic-moe/ablation-skill-inference/outputs/*`
- empty-skill charts:
  - [validation_baseline](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/charts/empty_skills/validation_baseline)
  - [validation_nb_top2](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/charts/empty_skills/validation_nb_top2)
  - [profile_pool](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/charts/empty_skills/profile_pool)
- per-expert profile distributions:
  - [dynahate_hate_skill_distribution.png](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/charts/profiles/dynahate_hate_skill_distribution.png)
  - [tweeteval_offense_skill_distribution.png](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/charts/profiles/tweeteval_offense_skill_distribution.png)
  - [kaggle_bully_skill_distribution.png](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/charts/profiles/kaggle_bully_skill_distribution.png)
  - [jigsaw_threat_skill_distribution.png](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/charts/profiles/jigsaw_threat_skill_distribution.png)
- top-k calibration:
  - [nb_topk_calibration.md](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/results/nb_topk_calibration.md)

### Appendix E: Qualitative Routing Case Studies

Recommended appendix-only content:

- successful routed examples
- failed routed examples
- empty-skill examples
- baseline vs NB contrast examples

Use the candidate case IDs already listed in [thesis_appendix_draft.md](/home/mb05005/mudit/llama_dora/LLaMA-Factory/symbolic-moe/thesis_appendix_draft.md).

### Appendix F: Artifact Map and Reproduction Paths

Keep these as appendix reference material:

- artifact inventory tables
- routed output variant table
- reproduction order list

## Items to Remove from the Older Outline

These no longer fit the current thesis:

- threshold sweep figure over `alpha`
- threshold sweep table over `alpha`
- any main-text figure that assumes the old positive-score / threshold router is still the final method
- any results table that treats the old alpha calibration as the selected operating point

Replace them with:

- Bernoulli top-k calibration table
- gold-expert recall vs top-k figure
- routing tradeoff discussion centered on `top-1/top-2/top-3`

## Recommended Minimal Main-Text Set

If you need a compact final thesis set, keep these in the main body:

- Figure 4.1 pipeline
- Figure 4.2 skill-inference workflow
- Figure 5.1 dataset skill heatmap
- Figures 5.2-5.5 per-dataset skill distributions
- Figure 5.6 expert profile heatmap
- Table 4.1 dataset summary
- Table 4.2 data pool composition
- Table 4.3 expert summary
- Table 4.4 shared training hyperparameters
- Table 4.5 dataset-specific training settings
- Table 4.6 skill vocabulary
- Tables 5.2-5.5 benchmark comparisons
- Table 5.6 profile statistics summary
- Table 5.7 main validation routing results
- Table 5.8 main test routing results
- Table 5.9 top-k calibration summary
- Figure 5.11 gold-expert recall at top-k

Everything denser than that can move to the appendix.
