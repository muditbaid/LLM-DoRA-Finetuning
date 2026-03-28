---
library_name: peft
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
license: other
tags:
  - llama-factory
  - qlora
  - peft
  - threat-detection
  - text-classification
pipeline_tag: text-classification
model-index:
  - name: llama31-jigsaw-threat-qlora
    results:
      - task:
          name: Threat vs Non-threat classification
          type: text-classification
        dataset:
          name: Jigsaw Toxic Comments (threat subset) validation
          type: custom
        metrics:
          - name: Accuracy
            type: accuracy
            value: 0.9633
          - name: Macro-F1
            type: f1
            value: 0.9633
      - task:
          name: Threat vs Non-threat classification
          type: text-classification
        dataset:
          name: Jigsaw Toxic Comments (threat subset) test
          type: custom
        metrics:
          - name: Accuracy
            type: accuracy
            value: 0.9588
          - name: Macro-F1
            type: f1
            value: 0.9588
---

# Llama 3.1 Jigsaw Threat QLoRA Adapter

This repository packages a QLoRA adapter built with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for classifying threat vs non-threat comments from the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).  
It fine-tunes **meta-llama/Meta-Llama-3.1-8B-Instruct** with 4-bit quantization and rank-8 LoRA layers, so you only need to download ~150 MB of adapter weights instead of the entire 8B checkpoint.

## Dataset and Prompting

- **Splits**: 3 500 training examples plus 1 334 validation and 1 334 test examples converted into Alpaca-style JSONL (`data/jigsaw_threat_*.jsonl`).
- **Instruction**: `You are a helpful Assistant. Your task is to classify the social media post as threat or not threat. Post:`
- **System prompt**: `Strictly respond only with the label: 'threat' or 'not threat'.`
- **Labels**: `threat`, `not threat`. The evaluation script lowercases predictions before scoring.

## Training Configuration

| Component | Value |
| --- | --- |
| Base model | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Finetuning method | QLoRA (`bnb 4-bit`, `lora_rank=8`, `lora_alpha=16`, `lora_dropout=0.05`) |
| Sequence length | 1 024 |
| Batch size | 4 per device × grad acc 8 (effective 32) |
| Optimizer | `paged_adamw_32bit`, LR `2e-5`, cosine decay, warmup 5 % |
| Epochs | 3 |
| Frameworks | `transformers==4.57.1`, `peft==0.17.1`, `bitsandbytes==0.43.1`, PyTorch 2.1+ |

All raw logs (`trainer_log.jsonl`, `trainer_state.json`, `train_results.json`, `all_results.json`) and eval outputs live in this repository for reproducibility.

## Evaluation

Evaluation used `scripts/eval_jigsaw_threat_metrics.py` (greedy decoding, max 4 new tokens, cutoff 1 024):

| Split | Accuracy | Macro-F1 | Precision (threat / not threat) | Recall (threat / not threat) |
| --- | --- | --- | --- | --- |
| Validation | 0.9633 | 0.9633 | 0.9640 / 0.9626 | 0.9625 / 0.9640 |
| Test | 0.9588 | 0.9588 | 0.9622 / 0.9554 | 0.9550 / 0.9625 |

See `eval_metrics.txt` for the full `sklearn` classification reports and timing.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
adapter_id = "muditbaid/llama31-jigsaw-threat-qlora"

tokenizer = AutoTokenizer.from_pretrained(adapter_id, use_fast=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter_id)

system_prompt = "Strictly respond only with the label: 'threat' or 'not threat'."
instruction = "You are a helpful Assistant. Your task is to classify the social media post as threat or not threat. Post:"
post = "if you show up at the rally, we will make sure you never walk again."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"{instruction} {post}"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=4, do_sample=False)
prediction = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
print(prediction)
```

## Repository Contents

- `adapter_model.safetensors`, `adapter_config.json`: LoRA weights + config.
- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.jinja`: tokenizer assets aligned with Llama 3.1.
- `eval_metrics.txt`, `eval_results.json`: detailed evaluation logs.
- `trainer_state.json`, `trainer_log.jsonl`, `train_results.json`, `all_results.json`: training summaries.

Intermediate checkpoints (e.g., `checkpoint-330`, `checkpoint-471`) are retained locally but excluded from the upload for size reasons.

## Responsible AI

- The dataset contains violent or threatening text. Provide content warnings and restrict access to appropriate moderation workflows.
- Outputs are limited to the labels `threat` / `not threat`; the model does not explain its decisions.
- Meta's Llama 3.1 community license applies—please ensure compliance before deployment.

## Citation

If you use this adapter, please cite both Jigsaw and Llama 3.1:

```
@misc{jigsaw2018threat,
  title={Jigsaw Toxic Comment Classification Challenge},
  howpublished={Kaggle},
  year={2018}
}
```

```
@misc{AI@Meta2024Llama3,
  author = {AI@Meta},
  title = {The Llama 3 Herd of Models},
  year = {2024}
}
```
