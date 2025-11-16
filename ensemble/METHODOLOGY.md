# Moderation Ensemble Methodology

This document summarizes how we (1) train and evaluate the individual “expert” adapters for hate/offense/bullying detection and (2) layer a Symbolic‑MoE–style routing pipeline on top so we can deliver consistent, multi-label moderation decisions at inference time.

---

## 1. Expert Adapters

### 1.1 Datasets

| Expert | Dataset | Labels | Notes |
|--------|---------|--------|-------|
| Hate/offense | HateXplain (`data/hatexplain_*.jsonl`) | `hatespeech`, `offensive`, `normal` | Balanced 3-class classification framed as single-label generation. |
| Bullying | Kaggle Cyberbullying (`data/kaggle_cyberbullying_*.jsonl`) | `label: bully/not_bully`, `type: age/gender/ethnicity/religion/none` | Structured output string with high-level + subtype. |
| Optional future experts | e.g., DynaHate, Dynabench toxicity, Dynaboard hate, etc. | Domain-specific binary labels | Plugs into the same interface later. |

Each dataset keeps validation splits so we can generate gold metrics and later build routing profiles.

### 1.2 Training

We fine-tune QLoRA adapters on Meta Llama‑3.1‑8B-Instruct using the provided `examples/train_qlora/*.yaml` configs:

1. `llama31_hatexplain_qlora_sft.yaml` → Hate/offense expert.
2. `llama31_kaggle_cyberbullying_qlora_sft.yaml` → Bullying + subtype expert.
3. Additional configs (e.g., `llama31_dynahate_qlora_sft.yaml`) follow the same recipe for future experts.

Key training details:

- LoRA rank/alpha tuned per task.
- Chat template = Llama‑3 system/user/assistant format.
- Supervised fine-tuning target is **exact label text** (“hatespeech”, “label: bully; type: age”, etc.).
- Validation runs produce per-label F1/accuracy stored in each adapter directory (e.g., `saves/.../eval_metrics.json`).

### 1.3 Evaluation Scripts

For reproducible metrics we keep task-specific evaluation CLIs:

- `scripts/eval_hatexplain_metrics.py` – single-label accuracy + macro-F1.
- `scripts/eval_cyberbullying_metrics.py` – label accuracy, subtype accuracy, macro-F1, joint correctness.
- `scripts/eval_dynahate_metrics.py` (and future ones) – shaped to each dataset’s schema.

These scripts:

1. Rebuild the Llama‑3 chat prompt.
2. Run deterministic generation (no sampling) so evaluation is stable.
3. Parse outputs with regex to extract the label(s).
4. Compare against gold labels to compute task metrics.

Outputs (JSON + console) prove each adapter is trustworthy on its specialization before we attempt any ensemble.

---

## 2. Symbolic-MoE Style Routing

We adopt the Symbolic Mixture-of-Experts idea (Chen et al., 2025) but tailor it to moderation:

### 2.1 Skill/Keyword Extraction

Purpose: infer which *types* of moderation labels might apply to a new post before running every expert.

Implementation options:

1. **Keyword LLM** – a lightweight instruction model (e.g., Qwen2.5‑7B-Instruct or Llama‑3.1‑8B) that, given the post, emits a short list of tags from our taxonomy (`["hate", "offense", "bully_age", ...]`). We prompt it 3‑5 times per example during profiling and keep tags that repeat for stability.
2. **Heuristic/Classifier** – if we have pre-existing detectors for “slur present” or “mentions age”, we can treat those as skills directly.

### 2.2 Profiling Each Expert (Offline)

Using a validation set large enough to cover every tag:

1. Tag each validation post with skills using the method above.
2. Run **each** expert adapter once per post to get predictions.
3. For every skill attached to a post, update the expert’s score: +1 if the expert got the gold label correct for its task, −1 otherwise.
4. The result is a dictionary per expert, e.g.:
   ```json
   {
     "hate": 85,
     "offense": 42,
     "normal": 60,
     "bully_age": -5,
     "bully_gender": -12,
     "...": 0
   }
   ```
5. Also record a **global competency** (sum of positive scores) to bias routing toward consistently reliable models.

We store these profiles under `ensemble/artifacts/` (or another JSON) so inference only needs a lookup.

### 2.3 Routing at Inference

For a new post:

1. Run the keyword extractor to get its skill list `K_post`.
2. For each expert `E_i`, compute a **local suitability** score by summing the expert’s profile entries for `K_post`.
3. Multiply by the expert’s global competency (normalized) to get a relevance weight.
4. Apply softmax (temperature ≈ 0.5) over weights and **threshold or sample** the top‑k experts.
5. Run only the selected experts’ inference code (e.g., `score_labels_from_logits`).

Routing outputs:

- A list of `(expert_name, selected_skills, probability distribution / logits)`.
- No aggregator is necessary if we’re willing to show multiple labels side by side; we simply surface each expert’s calibrated probabilities and final decision.

### 2.4 Optional Aggregation (Future)

If we later decide to return a single final label/rationale, we can add Symbolic‑MoE’s aggregator step:

1. Build a synthetic aggregation benchmark (one correct + two incorrect CoTs) from validation data.
2. Evaluate each expert in “judge” mode to see which synthesizes others best.
3. For each task choose the best aggregator and use it to fuse the selected experts’ outputs.

For now we skip this and present multiple expert decisions explicitly.

### 2.5 Efficiency Considerations

- **Batching:** we batch all posts routed to the same expert so that model only loads once per batch. This mirrors Symbolic‑MoE’s trick that lets them run up to 16 experts on a single GPU.
- **Expert pruning:** during profiling we track how often each expert is selected; extremely low-frequency experts can be dropped or merged to lower latency.
- **Calibration & weights:** we can optionally apply the calibration/weighting pipeline (`ensemble/run_ensemble_eval.py`) to keep probabilities comparable across experts before presenting them.

---

## 3. Putting It Together

1. Train QLoRA adapters per dataset (Sec. 1). Save their checkpoints + eval metrics.
2. Run `ensemble/run_ensemble_eval.py` (once routing is ready) to gather logits, fit temperature scaling, tune thresholds, and collect masked metrics for transparency.
3. Build expert profiles + keyword extractor as described in Sec. 2.
4. At inference time:
   - Tag incoming post → skills.
   - Route to relevant experts using profiles.
   - Run selected experts and obtain calibrated probabilities / thresholds.
   - Emit structured output such as:
     ```json
     {
       "post": "...",
       "alerts": [
         {"expert": "hatexplain", "label": "hatespeech", "prob": 0.81, "threshold": 0.62},
         {"expert": "kaggle_bully", "label": "bully_age", "prob": 0.77, "threshold": 0.55}
       ]
     }
     ```
5. (Optional) add an aggregator/judge if we ever want one consolidated answer instead of multiple expert alerts.

This approach lets us keep each adapter specialized and trustworthy while still delivering richer moderation coverage—no multi-round discussions, no retraining, just smart routing on top of the experts we’ve already built.
