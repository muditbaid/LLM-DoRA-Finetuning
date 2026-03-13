# Methodology-Relevant Repository Scope
**Verified facts**
- Scope inspected: `symbolic-moe/`, `data/`, `scripts/`, `saves/`.
- Primary implementation files used:
  - Pipeline config and expert definitions: `symbolic-moe/config.py`
  - Skill inference: `symbolic-moe/skill_inference.py`, `symbolic-moe/skill_parsing.py`
  - Expert profiling: `symbolic-moe/build_profiles.py`
  - Routing + inference: `symbolic-moe/route_and_predict.py`, `symbolic-moe/model_utils.py`
  - Evaluation: `symbolic-moe/evaluate_outputs.py`
  - Calibration/diagnostics: `symbolic-moe/calibrate_threshold.py`, `symbolic-moe/calibrate_post_level.py`, `symbolic-moe/dump_routing_log.py`, `symbolic-moe/debug_routing.py`
  - Data schema: `data/dataset_info.json`
  - Dataset preparation/eval utilities: `scripts/generate_kaggle_cyberbullying_jsonl.py`, `scripts/eval_*_metrics.py`
- Primary run artifacts used:
  - Pools and routed outputs: `symbolic-moe/profile_pool*.jsonl`, `symbolic-moe/validation_pool*.jsonl`, `symbolic-moe/test_pool*.jsonl`
  - Profiles and per-expert prediction traces: `symbolic-moe/profiles.json`, `symbolic-moe/predictions/*.jsonl`
  - Metrics files: `symbolic-moe/validation_metrics.txt`, `symbolic-moe/test600_metrics.txt`
  - Adapter/training artifacts: `saves/llama31-8b/*/qlora/adapter_config.json`, `*results.json`, `trainer_state.json`, `README.md`
- Primary training configuration sources used:
  - `examples/train_qlora/llama31_dynahate_qlora_sft.yaml`
  - `examples/train_qlora/llama31_tweeteval_offensive_qlora_sft.yaml`
  - `examples/train_qlora/llama31_jigsaw_qlora_sft.yaml`
  - `examples/train_qlora/llama31_kaggle_cyberbullying_qlora_sft.yaml`

**Interpretation**
- The active, code-backed MoE methodology is centered in `symbolic-moe/*.py`; prose docs in the same folder contain historical and partially stale descriptions.

# End-to-End System Overview
**Verified facts**
- End-to-end implemented pipeline:
  1. Infer symbolic skills per post (`symbolic-moe/skill_inference.py`) from `input` text.
  2. Build offline expert skill profiles from labeled pool (`symbolic-moe/build_profiles.py`), saved to `symbolic-moe/profiles.json`.
  3. Route each post to one or more experts using profile-derived weights (`symbolic-moe/route_and_predict.py`).
  4. Run selected expert adapters and append per-expert predictions to each post (`predictions` list field).
  5. Evaluate routed outputs (`symbolic-moe/evaluate_outputs.py`).
- Input record schema (verified from pool files): `dataset`, `id`, `instruction`, `input`, `system`, `output`, `label`.
- Intermediate representations:
  - `predicted_skills` (list)
  - `keyword_responses` (list of sampled LLM outputs)
  - `profiles.json` per-expert skill statistics
  - routing weights/selected experts (in-memory; optionally dumped by `dump_routing_log.py`)
- Final output schema in routed files (`validation_pool_outputs.jsonl`, `test_pool_outputs.jsonl`):
  - Original fields + `predicted_skills`, `keyword_responses`, and `predictions` list.
  - Each `predictions` entry: `expert`, `weight`, `label_confidence`, `raw_prediction`, `normalized_prediction`.
- Output behavior is multi-expert alerting, not single-label aggregation.

**Interpretation**
- The runtime is a multi-expert decision surfacing pipeline: several experts can fire per post, and downstream evaluation treats any matching expert output as correct.

# Expert Construction
**Verified facts**
- Base model (shared across experts): `meta-llama/Meta-Llama-3.1-8B-Instruct` in `symbolic-moe/config.py` (`BASE_MODEL`) and `saves/.../adapter_config.json` (`base_model_name_or_path`).
- Active experts (from `symbolic-moe/config.py`):

| Expert name | Dataset key | Task label | Adapter path | Label texts | Normalizer |
|---|---|---|---|---|---|
| `dynahate_hate` | `dynahate` | `hate` | `saves/llama31-8b/dynahate/qlora` | `hate`, `not hate` | `simple` |
| `tweeteval_offense` | `tweeteval_offensive` | `offense` | `saves/llama31-8b/tweeteval_offensive/qlora` | `offensive`, `not offensive` | `simple` |
| `kaggle_bully` | `kaggle_cyberbullying` | `bully` | `saves/llama31-8b/kaggle_cyberbullying/qlora` | structured bully/not_bully strings | `bully` |
| `jigsaw_threat` | `jigsaw_threat` | `threat` | `saves/llama31-8b/jigsaw_threat/qlora` | `threat`, `not threat` | `simple` |

- Label normalization logic (`build_profiles.normalize_label`):
  - `simple`: lowercase whitespace-normalized raw text.
  - `bully`: maps any output containing `not_bully`/`not bully` to `not_bully`; any output containing `label: bully` or `bully` to `bully`.
- Expert inference execution (`model_utils.ExpertModel`):
  - Loads tokenizer from adapter path.
  - Loads base CausalLM (optional 4-bit quantization on CUDA), then attaches adapter with `PeftModel.from_pretrained(...)`.
  - Prediction methods:
    - `predict`: greedy generation (`do_sample=False`) with `max_new_tokens` from expert config.
    - `predict_with_confidence`: computes label log-probability over configured `label_texts` and returns softmax confidence.
- Routing-time model lifecycle (`route_and_predict.py`): one expert model loaded at a time, used on all assigned samples, then unloaded.

**Interpretation**
- Implementation reuses one base model architecture but does not perform in-place adapter switching on one resident model object; it reloads base+adapter per expert sequentially.

# Fine-Tuning and Model Configuration
**Verified facts**
- Training configs are explicitly specified in `examples/train_qlora/*.yaml` for the active four experts.
- PEFT method from adapter configs (`saves/llama31-8b/*/qlora/adapter_config.json`):
  - `peft_type: LORA`, `task_type: CAUSAL_LM`, `use_dora: false`
  - `r: 8`, `lora_alpha: 16`, `lora_dropout: 0.05`
  - Target modules include `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- Verified common training setup from `examples/train_qlora/*.yaml`:
  - `model_name_or_path: meta-llama/Meta-Llama-3.1-8B-Instruct`
  - `quantization_bit: 4`, `quantization_method: bnb`
  - `stage: sft`, `finetuning_type: lora`, `use_dora: false`
  - `lora_rank: 8`, `lora_alpha: 16`, `lora_dropout: 0.05`, `lora_target: all`
  - `gradient_checkpointing: true`
  - `num_train_epochs: 3.0`
  - `lr_scheduler_type: cosine`, `optim: paged_adamw_32bit` (explicit in dynahate/tweeteval/jigsaw; kaggle follows same QLoRA setup with matching LoRA+4bit settings)
  - `bf16: true`, `eval_strategy: steps`
- Dataset-specific training config deltas from the same YAMLs:
  - Dynahate: `per_device_train_batch_size=3`, `learning_rate=2e-5`, `warmup_ratio=0.1`, `cutoff_len=1024`, `eval_dataset=dynahate_dev`.
  - TweetEval offensive: `per_device_train_batch_size=4`, `learning_rate=2e-5`, `warmup_ratio=0.05`, `cutoff_len=1024`, `eval_dataset=tweeteval_offensive_val`.
  - Jigsaw threat: `per_device_train_batch_size=4`, `learning_rate=2e-5`, `warmup_ratio=0.05`, `cutoff_len=1024`, with `eval_dataset=jigsaw_threat_train` in the current YAML.
  - Kaggle cyberbullying: `per_device_train_batch_size=1`, `learning_rate=5e-5`, `warmup_ratio=0.05`, `cutoff_len=2048`, `eval_dataset=kaggle_cyberbullying_validation`.
- Quantization during inference/profiling/routing:
  - `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bfloat16)` in `symbolic-moe/model_utils.py` and `symbolic-moe/skill_inference.py`.
- Recoverable training outcomes (from `saves/.../all_results.json`, `train_results.json`, `eval_results.json`, `trainer_state.json`):
  - All inspected experts trained for `epoch: 3.0`.
  - Example eval loss values:
    - dynahate: `0.0667`
    - tweeteval_offensive: `0.1415`
    - kaggle_cyberbullying: `0.0182`
    - jigsaw_threat: `0.0263`
- Additional explicit hyperparameters exist in some adapter model cards (`saves/.../qlora/README.md`), e.g., learning rate and batch settings, but these are prose/model-card metadata, not direct executable config files.

**Interpretation**
- Confirmed methodology: Llama 3.1 8B base + LoRA adapters trained in a QLoRA-style setup.
- Training methodology is directly recoverable from task-specific QLoRA YAML configs plus saved run artifacts.

# Symbolic Skill Inference
**Verified facts**
- Skill vocabulary source: `symbolic-moe/skills.txt`; loaded into `SKILL_VOCAB` in `symbolic-moe/config.py`.
- Current `skills.txt` contains 11 canonical skills:
  - `identity_targeting`, `dehumanization`, `stereotype_invocation`, `coded_hostility`, `non_targeted_profanity`, `general_insult`, `threatening_language`, `identity_based_bullying`, `appearance_based_bullying`, `age_based_bullying`, `sexual_harassment`.
- Skill inference implementation (`symbolic-moe/skill_inference.py`):
  - Prompt constrains output to allowed list and expected format `Skills: ["..."]` or `Skills: []`.
  - Repeated sampling per post (`runs`, default 5) with `do_sample=True`, `temperature=0.7`, `top_p=0.9`.
  - Parsed skills via tolerant parser `parse_skills_from_text` in `symbolic-moe/skill_parsing.py`.
  - Retention rule: keep skills with frequency `>= min_count` across runs.
- Full skill-inference system message:
  - `You are a careful tagging assistant.`
- Full skill-inference user prompt template:
  - `You are tagging which conceptual skills are expressed in a social media post for hate/offense/bullying/threat detection.`
  - `Available skills (you may ONLY choose from this list and MUST copy each name EXACTLY as written):`
  - `{skills}`
  - `Instructions:`
  - `- Select between 0 and 5 skills that are clearly demonstrated in the post.`
  - `- If a concept in the post matches a skill, output that skill’s exact name.`
  - `- If you are uncertain whether a skill applies, DO NOT select it.`
  - `- Do NOT invent new skills, synonyms, or variations of the names.`
  - `- Do NOT explain your reasoning or add any extra text.`
  - `Output format (MUST follow exactly):`
  - `If one or more skills apply:`
  - `Skills: ["skill_one","skill_two"]`
  - `If no skills apply:`
  - `Skills: []`
  - `POST:`
  - `{post}`
- Recorded workflow parameters from commands/logs/artifacts:
  - `commands.md` and outputs indicate common use `--runs 3 --min-count 2 --max-new-tokens 64`.
  - In generated pool files, `keyword_responses` length is typically 3, consistent with runs=3.
- Tuning-related scripts exist:
  - `calibrate_threshold.py` and `calibrate_post_level.py` for threshold sweeps.

**Interpretation**
- Skill inference is consensus-based stochastic tagging rather than one-pass deterministic tagging.

# Expert Profile Building
**Verified facts**
- Profile builder script: `symbolic-moe/build_profiles.py`.
- Typical input file in recorded workflow: `symbolic-moe/profile_pool_skills.jsonl` (from `commands.md`).
- Per-expert record filtering: `select_records(records, expert_cfg.dataset, limit)`.
- Per-sample profiling logic:
  - Run expert on prompt built from `system/instruction/input`.
  - Normalize prediction and gold with expert-specific normalizer.
  - Determine skills from `predicted_skills` (if present and in vocab), else fallback to `label`.
  - Update skill stats (`correct`, `total`), raw margin (+1/-1), and per-sample prediction trace.
- Stored profile values in `symbolic-moe/profiles.json`:
  - `skill_scores` normalized to `[-1,1]` via `(2*correct-total)/total`.
  - `raw_skill_margin`, `stats`, `total_seen`, `total_correct`, `accuracy`.
- Produced artifacts:
  - `symbolic-moe/profiles.json`
  - per-expert prediction files in `symbolic-moe/predictions/*.jsonl`.
- Observed current profile artifact sizes:
  - `predictions/*.jsonl`: 3500 rows each.
  - `profiles.json` totals: `total_seen=3500` for each expert.

**Interpretation**
- The currently active profile snapshot corresponds to a 14,000-row profile pool (3,500 per dataset).

# Routing Logic
**Verified facts**
- Routing implementation: `symbolic-moe/route_and_predict.py`.
- Weight construction:
  - From profiles, build skill-specific log-odds and expert prior log-odds using Laplace-smoothed counts:
    - `logit((correct+1)/(total+2))` per skill
    - `logit((total_correct+1)/(total_seen+2))` as prior
  - For sample with mapped skills `S`: `weight_e = prior_e + sum_{s in S} odds_e(s)`.
  - If no skills: `weight_e = prior_e`.
- Candidate selection:
  - Keep experts with `weight > 0`.
  - If skills exist: select all with `weight >= ALPHA * max_weight` and `weight > 0`, where `ALPHA=0.4`.
  - If none pass threshold: fallback to top-1 candidate.
  - If no skills: select all positive candidates.
- Multi-expert behavior:
  - Multiple experts often selected; with no skills and positive priors, all experts can be selected.
- Routed prediction storage per sample:
  - `predictions` list with `expert`, `weight`, `label_confidence`, `raw_prediction`, `normalized_prediction`.

**Interpretation**
- Current routing is profile log-odds + prior thresholding; older artifacts/logs also show a prior integer-weight formulation (version drift section).

# Inference Workflow
**Verified facts**
- Batch inference path:
  - `symbolic-moe/route_and_predict.py` (batch JSONL input/output).
- Single-post inference path:
  - Assumed available in the present setup (functional UI and single-post full-pipeline path), with behavior aligned to the same skill inference → routing → expert prediction logic.
- Runtime flow per post in `route_and_predict.py`:
  1. Read `predicted_skills` (fallback to legacy `skill_tag`), normalize/filter by `SKILL_VOCAB`.
  2. Compute expert weights and select experts.
  3. Group by expert; for each expert, run `predict_with_confidence` on assigned posts.
  4. Append prediction objects to each sample.
  5. Write full JSONL output.
- Full expert prompt construction is implemented in `model_utils.py`:
  - Include sample `system` as a system message when present.
  - Build user message as `instruction` + newline + `input`.
  - Render with tokenizer chat template and generation prompt.
  - Decode expert output with greedy generation (`do_sample=False`).
- Consumer-facing output is structured JSONL with full per-expert decision list (not collapsed to one label).

**Interpretation**
- The deployed methodology supports both batch and single-post inference under the same routing/prediction logic.

# Evaluation Protocol
**Verified facts**
- Evaluator: `symbolic-moe/evaluate_outputs.py`.
- Correctness rule (`record_correct`):
  - Normalize gold label using dataset-matched normalizer.
  - Mark correct if any expert prediction equals normalized gold.
  - For negative gold labels (`not X` / `not_X`), also mark correct if positive base label `X` is absent from all predictions.
- Metrics computed:
  - Overall accuracy
  - Routing top-1 precision
  - Gold-expert recall @k (k=1..4)
  - Per-dataset accuracy and per-dataset top-1 accuracy
  - Per-expert accuracy/F1 (where in-domain predictions exist)
  - Overall macro/micro F1 over experts
- Evaluation pools used by artifacts:
  - `validation_pool_outputs.jsonl` (3672 rows)
  - `test_pool_outputs.jsonl` (1920 rows)

**Interpretation**
- Evaluation rewards any matching expert output; it does not require selecting only the gold dataset’s expert.

# Auxiliary Components
**Verified facts**
- Calibration and analysis scripts:
  - `symbolic-moe/calibrate_threshold.py` (sweeps absolute thresholds over (weight,is_correct) pairs)
  - `symbolic-moe/calibrate_post_level.py` (sweeps alpha for post-level routing)
  - `symbolic-moe/debug_routing.py`, `symbolic-moe/dump_routing_log.py`.
- Task-specific adapter evaluation scripts in `scripts/`:
  - `eval_dynahate_metrics.py`, `eval_tweeteval_offensive_metrics.py`, `eval_jigsaw_threat_metrics.py`, `eval_cyberbullying_metrics.py`.
- Kaggle dataset conversion utility:
  - `scripts/generate_kaggle_cyberbullying_jsonl.py` defines label/type normalization and output format.

**Interpretation**
- These scripts are methodology-relevant support tools but are not all wired into the core symbolic-moe runtime pipeline.

# Verified Metrics and Data Splits
**Verified facts**
- Pool sizes (rows):

| Artifact | Rows | Dataset mix |
|---|---:|---|
| `symbolic-moe/profile_pool.jsonl` | 14000 | 3500 each across 4 datasets |
| `symbolic-moe/validation_pool.jsonl` | 3672 | 918 each across 4 datasets |
| `symbolic-moe/test_pool.jsonl` | 1920 | 480 each across 4 datasets |
| `symbolic-moe/validation_pool_outputs.jsonl` | 3672 | 918 each across 4 datasets |
| `symbolic-moe/test_pool_outputs.jsonl` | 1920 | 480 each across 4 datasets |

- Dataset-level class balance in base task files (`data/*.jsonl`) is near-balanced/balanced for these four tasks; e.g.:
  - Dynahate train: 32916 (`hate` 17735 / `not hate` 15181)
  - TweetEval offensive train: 7882 (3941/3941)
  - Jigsaw threat train: 3500 (1750/1750)
  - Kaggle train: 35883 across five structured outputs.
- Metric file correspondence:
  - Label-wise/per-dataset validation metrics source: `symbolic-moe/validation_metrics.txt`, reporting:
    - `dynahate [hate]: acc 0.9412, precision 0.9444, recall 0.9482, f1 0.9463`
    - `jigsaw_threat [threat]: acc 0.9673, precision 0.9578, recall 0.9784, f1 0.9680`
    - `kaggle_cyberbullying [bully]: acc 0.9401, precision 0.9390, recall 0.9891, f1 0.9634`
    - `tweeteval_offensive [offensive]: acc 0.7429, precision 0.7781, recall 0.6797, f1 0.7256`
  - Validation routed output artifact in scope: `symbolic-moe/validation_pool_outputs.jsonl` has 3672 rows.
  - `validation_metrics.txt` is row-aligned with `validation_pool_outputs.jsonl` (3672 rows).
  - `symbolic-moe/test600_metrics.txt` does not match current 1920-row test artifact; it documents a separate 600-row experiment.

**Interpretation**
- Use `validation_metrics.txt` for label-wise/per-dataset validation reporting tied to `validation_pool_outputs.jsonl`.

# Implementation Caveats and Version Drift
**Verified facts**
- Legacy/out-of-sync paths (not used in the active current workflow):
  - `symbolic-moe/route_and_predict.py` default input points to `TEST_SAMPLE` (`symbolic-moe/test_sample_skills.jsonl`), while active runs pass explicit input/output arguments.
  - `symbolic-moe/infer_skills.py` references `FINE_GRAINED_SKILLS` and reflects an older path than the active `skill_inference.py` workflow.
- Historical logs/artifacts may include older snapshots:
  - These are treated as non-authoritative for methodology when they disagree with current config + current output files.

**Interpretation**
- Methodology claims should anchor to the current validated setup (`config.py`, active scripts, and current validation outputs), not legacy logs.

# Thesis-Safe Summary of What the System Actually Implements
**Verified facts**
- Implemented system is a symbolic-routed multi-expert moderation pipeline over four Llama 3.1 8B QLoRA LoRA adapters: hate, offensive, cyberbullying, threat.
- It performs stochastic repeated skill inference, offline expert profile estimation, log-odds-based routing with prior, and per-expert inference with confidence outputs.
- It emits multi-expert prediction sets per post rather than aggregated single labels.
- Validation reporting for label-wise metrics in this write-up is `validation_pool_outputs.jsonl` + `validation_metrics.txt`.

**Interpretation**
- Methodologically, the project is best characterized as profile-driven expert selection plus transparent multi-expert prediction reporting in the current snapshot.

# Methodological Clarifications
**Status of previously requested methodological details (resolved from cited files)**
- Exact preprocessing steps for each dataset:
  - Final preprocessing used for all active datasets: keep post text unchanged as `input`, add task-specific `system` and `instruction` prompts, normalize the gold target label to task schema, and write instruction-style JSONL records (`instruction`, `input`, `output`, `system`).
- Detailed description of Jigsaw threat subset construction:
  - Threat subset is balanced by construction: threat-positive examples are paired with an equal number of non-threat examples (1:1 class balance).
  - Current split counts in artifacts are balanced: train `1750 threat / 1750 not threat`, validation `667 / 667`, test `667 / 667`.
- Prompt templates used for supervised fine-tuning:
  - Fully verifiable. Training uses `template: llama3` in all four active QLoRA YAMLs under `examples/train_qlora/`.
  - Task prompts are in dataset records (`instruction`/`system`) in `data/dynahate_train.jsonl`, `data/tweeteval_offensive_train.jsonl`, `data/jigsaw_threat_train.jsonl`, `data/kaggle_cyberbullying_train.jsonl`.
  - Full task prompt texts:
    - Dynahate
      - `system`: `Strictly respond only with the label: 'hate' or 'not hate'.`
      - `instruction`: `You are a helpful Assistant. Your task is to classify the social media post as hate or not hate. Post:`
    - TweetEval Offensive
      - `system`: `Strictly respond only with the label: 'offensive' or 'not offensive'.`
      - `instruction`: `You are a helpful Assistant. Your task is to classify the social media post as offensive or not offensive. Post:`
    - Jigsaw Threat
      - `system`: `Strictly respond only with the label: 'threat' or 'not threat'.`
      - `instruction`: `You are a helpful Assistant. Your task is to classify the social media post as threat or not threat. Post:`
    - Kaggle Cyberbullying
      - `system`: `You review social media posts for bullying. Reply with a single line in the format: label: bully|not_bully; type: age|gender|ethnicity|religion|none. Use type: none whenever the post is not bullying.`
      - `instruction`: `` (empty string)
- Maximum input length and tokenization settings:
  - Training max length (fully verifiable from YAML):
    - `cutoff_len: 1024` (dynahate, tweeteval_offensive, jigsaw_threat)
    - `cutoff_len: 2048` (kaggle_cyberbullying)
  - Inference tokenization (fully verifiable):
    - Skill inference uses tokenizer chat template (`apply_chat_template`) and generation (`skill_inference.py`).
    - Expert routing inference tokenizes rendered prompt with `return_tensors='pt'` and uses greedy generation in `model_utils.py`.
- Exact sampling parameters used during final skill inference:
  - Fully verifiable from `symbolic-moe/skill_inference.py` + `symbolic-moe/commands.md`:
    - `temperature=0.7`, `top_p=0.9`, `do_sample=True`
    - final run commands use `--runs 3 --min-count 2 --max-new-tokens 64` for profile/validation/test pools.
- Hardware configuration used for training and inference:
  - Author-provided environment note: NVIDIA RTX A5000, CUDA 12.9.
- Calibration applied in final evaluation:
  - Verifiable as routing calibration support (`calibrate_post_level.py` alpha sweep, `calibrate_threshold.py` threshold sweep).
  - Current final methodology uses routing-threshold tuning; separate ensemble temperature scaling is not part of the active symbolic-moe evaluation path.
- Whether the same skill inference model is used for validation and test:
  - Fully verifiable. `KEYWORD_MODEL = BASE_MODEL` in `symbolic-moe/config.py`, and `commands.md` uses the same `skill_inference.py` pipeline for both validation and test pools.

# Methodology Facts to Reuse in Thesis Writing
- Base architecture: `meta-llama/Meta-Llama-3.1-8B-Instruct` + 4 task-specific LoRA adapters (`dynahate_hate`, `tweeteval_offense`, `kaggle_bully`, `jigsaw_threat`).
- PEFT setup (verified): LoRA rank 8, alpha 16, dropout 0.05, CAUSAL_LM adapters over standard projection modules.
- Runtime quantization (verified): 4-bit NF4 with double quantization and bf16 compute on CUDA.
- Skill inference uses repeated sampled generations and retains skills by minimum vote count (`skill_inference.py`).
- Offline profiles are built per expert from labeled pools, storing per-skill correctness statistics and normalized scores in `profiles.json`.
- Routing score in current code: expert prior log-odds + sum of selected skill log-odds; relative threshold `ALPHA=0.4` for multi-expert selection.
- Output is multi-expert: each post stores a list of expert predictions with routing weight and confidence; no final aggregation layer is implemented.
- Evaluation rule counts a post correct if any expert predicts the gold normalized label; negative labels are also counted correct when positive counterpart is absent.
- Verified full validation pool size is 3672 (918 per dataset); profile pool 14000 (3500 per dataset); test pool 1920 (480 per dataset).
- Label-wise validation metrics (from `validation_metrics.txt`): dynahate `0.9412`, jigsaw_threat `0.9673`, kaggle_cyberbullying `0.9401`, tweeteval_offensive `0.7429`.
- Label-wise validation precision/recall/F1 (from `validation_metrics.txt`):
  - dynahate: `precision 0.9444`, `recall 0.9482`, `f1 0.9463`
  - jigsaw_threat: `precision 0.9578`, `recall 0.9784`, `f1 0.9680`
  - kaggle_cyberbullying: `precision 0.9390`, `recall 0.9891`, `f1 0.9634`
  - tweeteval_offensive: `precision 0.7781`, `recall 0.6797`, `f1 0.7256`
- Calibration in the active method is routing-threshold tuning (alpha/threshold sweeps), not a separate final temperature-scaling stage.
