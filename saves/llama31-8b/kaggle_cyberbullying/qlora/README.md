---
library_name: peft
license: other
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
tags:
- base_model:adapter:meta-llama/Meta-Llama-3.1-8B-Instruct
- llama-factory
- lora
- transformers
- cyberbullying-detection
- content-moderation
pipeline_tag: text-generation
model-index:
- name: llama3.1-Instruct-qlora-cyberbullying
  results:
  - task:
      type: text-classification
      name: Cyberbullying Moderation (label + type)
    dataset:
      name: kaggle_cyberbullying_validation
      type: custom
      config: kaggle_cyberbullying_validation
      split: validation
    metrics:
    - type: accuracy
      value: 0.9511
    - type: recall
      name: macro
      value: 0.9511
    - type: precision
      name: macro
      value: 0.9507
    - type: f1
      name: macro
      value: 0.9509
---

# Llama 3.1 Instruct QLoRA – Cyberbullying Classifier

LoRA adapter for moderating social media posts. The model takes a post and returns a structured verdict in the format:

```
label: bully|not_bully; type: age|gender|ethnicity|religion|none
```

If bullying is detected, the adapter also predicts the bullying subtype; otherwise `type: none` is returned.

This adapter was fine-tuned from [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) using QLoRA with LLaMA-Factory. Training data is derived from [Kaggle’s Cyberbullying Classification dataset](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification).

## TL;DR

- **Task**: binary bully/not_bully classification + bullying subtype (age, gender, ethnicity, religion)
- **Training set**: 35,883 posts
- **Validation set**: 3,986 posts
- **Prompt**: single system + instruction message with deterministic output template (see below)
- **Result**: 95.1% joint label/type accuracy on validation

## Prompt template

**System**

> You are a careful, consistent content‑moderation assistant. First decide whether the post is bullying; if it is, assign the most specific bullying type (age, gender, ethnicity, or religion). If it is not bullying, use type: none. Always output a single line exactly in this format: label: <label>; type: <type>, with no additional text or punctuation.

**Instruction**

> Classify the following social media post. Return exactly one line in the format: label: bully|not_bully; type: <age|gender|ethnicity|religion|none>. Use lowercase exactly as shown. If the post is not bullying, set type to none. Do not include any extra words, explanations, or whitespace before or after the answer.

## Training configuration

- Framework: LLaMA-Factory (QLoRA stage)
- Base model: Meta-Llama-3.1-8B-Instruct
- Adapter type: LoRA (rank 8, alpha 16, dropout 0.05) — DoRA disabled for QLoRA
- Quantization: 4-bit bnb
- Cutoff length: 2,048 tokens
- Optimizer: AdamW (fused)
- Learning rate: 5e-5, cosine scheduler, warmup ratio 0.05
- Batch size: 1 sample × 8 grad accumulation (effective 8)
- Epochs: 3 (seed 42)
- Precision: bfloat16

Validation loss reached **0.018** by the end of training (see `trainer_state.json`).

## Evaluation results

Validation split: 3,986 posts (10% stratified sample, same prompt formatting).

| Type      | Precision | Recall | F1    | Support |
|-----------|-----------|--------|-------|---------|
| age       | 0.9899    | 0.9861 | 0.9880 | 793 |
| gender    | 0.9254    | 0.9279 | 0.9267 | 749 |
| ethnicity | 0.9858    | 0.9905 | 0.9881 | 839 |
| religion  | 0.9504    | 0.9677 | 0.9590 | 773 |
| none      | 0.9018    | 0.8834 | 0.8925 | 832 |
| **Accuracy** | —       | —      | **0.9511** | 3,986 |
| **Macro avg** | **0.9507** | **0.9511** | **0.9509** | 3,986 |
| **Weighted avg** | **0.9509** | **0.9511** | **0.9509** | 3,986 |

Joint label+type accuracy is therefore 95.1%. The adapter maintains high recall for protected classes and is intentionally conservative on the `none` category (non-bullying content).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "meta-llama/Meta-Llama-3.1-8B-Instruct"
adapter = "muditbaid/llama3.1-Instruct-qlora-cyberbullying"

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto")
model = PeftModel.from_pretrained(model, adapter)

system = ("You are a careful, consistent content-moderation assistant..." )  # see full text above
instruction = ("Classify the following social media post..." )
post = "Why are you acting like such a clown, grandma?"

prompt = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + system +
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + instruction + "\n" + post +
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### LLaMA-Factory usage

Add this repository as `adapter_path` in a LLaMA-Factory config (or pass `--adapter_path`) to evaluate or continue training.

## Dataset notes

- Source: Kaggle Cyberbullying dataset (5 classes: age, gender, ethnicity, religion, not_cyberbullying)
- Pre-processing: normalization to `label` ∈ {bully, not_bully}; map `not_cyberbullying → none` subtype; remove rows with missing text
- Split: 90/10 stratified on `class_label`

## Limitations

- English-only and trained on informal social media text
- Model may reproduce harmful language when prompted adversarially
- Designed for decision support; keep a human reviewer in the loop

## Citation

```
@misc{muditbaid2024llama31cyberbullying,
  title  = {Llama 3.1 Instruct QLoRA – Cyberbullying Classifier},
  author = {Baid, Mudit},
  year   = {2024},
  howpublished = {Hugging Face Hub},
  url    = {https://huggingface.co/muditbaid/llama3.1-Instruct-qlora-cyberbullying}
}
```

---

Trained with **PEFT 0.17.1**, **Transformers 4.57.1**, **PyTorch 2.9.0+cu128**, **Datasets 4.0.0**, **Tokenizers 0.22.1**.
