# 3 Methodology

## 3.1 Methodological Overview
Harmful-language detection in social media spans multiple related but non-identical phenomena, including hate speech, offensive language, cyberbullying, and threats. A single monolithic classifier can underperform when label definitions, prompt styles, and class boundaries differ across tasks. The methodology therefore adopts a symbolic mixture-of-experts design in which each expert model is specialized for one task while a symbolic routing layer decides which experts to activate per input.

The pipeline operates in three stages. First, a symbolic skill inference module extracts a compact set of interpretable cues from each post. Second, an offline profiling stage estimates expert competence conditioned on those skills. Third, a router combines skill-conditioned evidence with expert prior competence to select one or more experts for prediction. The system is intentionally multi-expert and multi-label in output behavior: it returns all selected expert predictions with confidence scores, rather than collapsing them into a single aggregated label.

The primary methodological contribution of this work is a symbolic mixture-of-experts architecture that integrates interpretable skill extraction with expert model routing. Instead of relying on a single classifier, the system first infers a set of symbolic skills from each post and then uses these signals to dynamically select the most appropriate expert models. This design enables specialization across related harmful-language detection tasks while maintaining interpretability in the routing decisions.

## 3.2 Task Formulation

The goal of the system is to detect harmful language in social media posts across multiple related categories. Let 
𝑥
x denote an input social media post represented as raw text. The system maps each post to one or more harmful-language labels using a set of specialized expert models.

Input
A single social media post 
𝑥
x, represented as unstructured text.

Output
A predicted label from a task-specific label space corresponding to the expert model activated by the routing system. The system may produce multiple expert predictions for the same post.

Tasks
The methodology addresses four related but distinct harmful-language detection tasks:

Hate speech detection

Offensive language detection

Cyberbullying detection

Threat detection

Each task is associated with a dedicated expert model trained on a task-specific dataset. Because definitions and annotation schemes differ across datasets, the system uses a symbolic routing mechanism to dynamically select the most appropriate expert models for each input.

## 3.3 Datasets
The implemented system uses four task-aligned datasets, each mapped to one expert:

1. DynaHate for hate vs. not-hate classification.
2. TweetEval Offensive for offensive vs. not-offensive classification.
3. Jigsaw Threat for threat vs. not-threat classification.
4. Kaggle Cyberbullying for bully vs. not-bully classification, with subtype annotations (age, gender, ethnicity, religion, none).

All datasets are represented in instruction-style JSONL records with system, instruction, input, and output fields. For the first three tasks, outputs are binary textual labels. For cyberbullying, outputs follow a structured label format (`label: ...; type: ...`).

Dataset preprocessing follows the same conversion pattern across tasks: post text is kept unchanged as `input`, task-specific `system` and `instruction` prompts are added, the gold target label is normalized to the task output schema, and the final record is saved as instruction-style JSONL.

For Jigsaw Threat, the threat subset is constructed as a balanced binary set by taking threat-positive posts and sampling an equal number of non-threat posts (1:1 class balance). This yields split-level balance in the current artifacts (train 1750/1750, validation 667/667, test 667/667).

## 3.4 Expert Model Construction
Each expert is instantiated as a LoRA adapter over a shared base language model, Meta-Llama-3.1-8B-Instruct. The active expert set comprises four adapters:

1. Hate expert (DynaHate).
2. Offensive-language expert (TweetEval Offensive).
3. Cyberbullying expert (Kaggle Cyberbullying).
4. Threat expert (Jigsaw Threat).

At inference time, the same base architecture is reused and expert-specific adapters are loaded for routed prediction. Expert outputs are normalized into task-specific canonical label spaces before evaluation.

## 3.5 Fine-Tuning Procedure
The fine-tuning method is supervised fine-tuning with QLoRA-style 4-bit quantization and LoRA adaptation. Verified common settings include:

1. Base model: Meta-Llama-3.1-8B-Instruct.
2. Quantization: 4-bit bitsandbytes.
3. PEFT method: LoRA (rank 8, alpha 16, dropout 0.05).
4. Training stage: supervised fine-tuning.
5. Gradient checkpointing enabled.
6. Three training epochs per expert.

The chat template used for training is `llama3`. Training examples are instruction-following classification prompts whose target text is the exact label string expected for each task. Maximum sequence length is task-dependent in the recovered configuration: 1024 tokens for DynaHate/TweetEval/Jigsaw and 2048 tokens for Kaggle Cyberbullying.

## 3.6 Symbolic Skill Inference
Before routing, each post is processed by a symbolic skill inference module. The module uses a fixed skill vocabulary and prompts a keyword model to output skills only from that controlled vocabulary. Skill inference is stochastic and repeated: multiple sampled generations are produced per post, then only skills that satisfy a minimum vote threshold across runs are retained.

In the validated workflow, the final settings are repeated sampling with temperature 0.7 and top-p 0.9, using three runs and minimum count two for retention. This consensus strategy reduces one-shot extraction variance and yields stable symbolic signals for downstream routing.

## 3.7 Expert Profiling
Expert profiling is performed offline on a profiled pool of labeled examples. For each expert, predictions are compared with gold labels after task-appropriate normalization. Skill-conditioned statistics are accumulated to estimate how reliable each expert is when particular symbolic skills are present.

The resulting profile stores:

1. Per-skill correctness statistics.
2. Per-skill normalized scores.
3. Global expert accuracy terms used as priors.

These profiles provide the empirical basis for routing decisions and are computed before final validation/test inference.

## 3.8 Routing Mechanism
Routing computes an expert relevance score by combining a prior competence term with skill-conditioned evidence. For expert \(e\) and inferred skill set \(S\), the implemented scoring rule is:

\[
\text{score}_e = \text{prior}_e + \sum_{s \in S} \text{odds}_{e,s}
\]

where both prior and skill terms are derived from Laplace-smoothed correctness statistics transformed to log-odds.

Experts with positive scores are candidate experts. With non-empty skill evidence, experts are selected if their score exceeds a relative threshold \(\alpha\) of the best candidate score; if none satisfy the threshold, the top expert is selected as fallback. In the active configuration, \(\alpha = 0.4\). The routing design permits multi-expert activation, consistent with the system’s multi-label alerting objective.

## 3.9 End-to-End Pipeline
Figure X.X illustrates the pipeline architecture. At runtime, processing follows:

1. Input post ingestion.
2. Symbolic skill inference.
3. Expert selection via profile-based routing.
4. Routed expert inference with confidence estimation.
5. Multi-expert output assembly.

The final output is a structured set of per-expert predictions and confidence scores for the same post. No post-hoc aggregation into a single label is applied in the current methodology.

## 3.10 Evaluation Protocol
Evaluation is performed on routed validation and test outputs. Each post may be routed to multiple experts based on the symbolic routing mechanism. To assess prediction correctness while accounting for multi-expert activation, evaluation follows a Top-2 expert rule.

For each post, experts are ranked by routing score. A prediction is considered correct if the gold normalized label is produced by any of the two highest-scoring routed experts.

This criterion balances strict routing evaluation with the system’s multi-expert design, where multiple experts may reasonably respond to the same input. Top-2 evaluation is commonly used in multi-expert or retrieval-style systems to measure whether the correct specialist model is among the most relevant candidates.

Reported metrics include overall accuracy and label-wise precision, recall, and F1 scores for each task. For the validation pool used in this methodology write-up, label-wise results are:

1. DynaHate (hate): accuracy 0.9412, precision 0.9444, recall 0.9482, F1 0.9463.
2. Jigsaw Threat (threat): accuracy 0.9673, precision 0.9578, recall 0.9784, F1 0.9680.
3. Kaggle Cyberbullying (bully): accuracy 0.9401, precision 0.9390, recall 0.9891, F1 0.9634.
4. TweetEval Offensive (offensive): accuracy 0.7429, precision 0.7781, recall 0.6797, F1 0.7256.