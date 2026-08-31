---
library_name: peft
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
license: other
tags:
  - llama-factory
  - qlora
  - peft
  - hatespeech-detection
  - text-classification
pipeline_tag: text-classification
model-index:
  - name: llama31-dynahate-qlora
    results:
      - task:
          name: Hate / Not Hate classification
          type: text-classification
        dataset:
          name: Dynahate dev
          type: custom
        metrics:
          - name: Accuracy
            type: accuracy
            value: 0.9259
          - name: Macro-F1
            type: f1
            value: 0.9255
      - task:
          name: Hate / Not Hate classification
          type: text-classification
        dataset:
          name: Dynahate test
          type: custom
        metrics:
          - name: Accuracy
            type: accuracy
            value: 0.9153
          - name: Macro-F1
            type: f1
            value: 0.9142
---

# Llama 3.1 Dynahate QLoRA Adapter

This repository hosts a QLoRA adapter built with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for classifying social media posts from the [Dynahate](https://aclanthology.org/2020.acl-main.633/) dataset into `hate` vs `not hate`.  
It fine-tunes **meta-llama/Meta-Llama-3.1-8B-Instruct** with 4-bit quantization and LoRA rank 8 adapters, so you only need to load ~150 MB of adapter weights instead of the full model checkpoint.

## Dataset and Prompting

- **Dynahate splits**: train (32 916), dev (4 100), test (4 120) CSVs converted into Alpaca-style JSONL (see `LLaMA-Factory/data/dynahate_*.jsonl` in the training workspace).
- **Instruction**: `You are a helpful Assistant. Your task is to classify the social media post as hate or not hate. Post:`
- **System prompt**: `Strictly respond only with the label: 'hate' or 'not hate'.`
- **Label space**: `hate`, `not hate`. Outputs are normalized case-insensitively during evaluation.

## Training Configuration

| Component | Value |
| --- | --- |
| Base model | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Finetuning method | QLoRA (bnb 4-bit, `lora_rank=8`, `lora_alpha=16`, `lora_dropout=0.05`) |
| Sequence length | 1 024 |
| Batch size | 3 per device × grad acc 8 (effective 24) |
| Optimizer | `paged_adamw_32bit`, LR `2e-5`, cosine schedule, warmup 10 % |
| Epochs | 3 |
| Framework versions | `transformers==4.57.1`, `peft==0.17.1`, `bitsandbytes==0.43.1`, PyTorch 2.1+ |

Full logs (training curves, trainer arguments, etc.) live inside this repository (`trainer_log.jsonl`, `trainer_state.json`, `all_results.json`).

## Evaluation

Evaluation was performed with `scripts/eval_dynahate_metrics.py` (greedy decoding, max 4 new tokens, cutoff 1 024).  
Detailed raw outputs are captured in `eval_metrics.txt`.

| Split | Accuracy | Macro-F1 | Precision (hate / not hate) | Recall (hate / not hate) |
| --- | --- | --- | --- | --- |
| Dynahate dev | 0.9259 | 0.9255 | 0.9228 / 0.9294 | 0.9382 / 0.9121 |
| Dynahate test | 0.9153 | 0.9142 | 0.9123 / 0.9191 | 0.9361 / 0.8898 |

The evaluation script also prints full `sklearn` classification reports (see `eval_metrics.txt`).

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
adapter_id = "muditbaid/llama31-dynahate-qlora"

tokenizer = AutoTokenizer.from_pretrained(adapter_id, use_fast=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, adapter_id)

instruction = "You are a helpful Assistant. Your task is to classify the social media post as hate or not hate. Post:"
system_prompt = "Strictly respond only with the label: 'hate' or 'not hate'."
post = "i can't stand those people coming into our neighborhood"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"{instruction} {post}"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=4, do_sample=False)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip())
```

## Files in this Repository

- `adapter_model.safetensors` / `adapter_config.json`: LoRA weights and configuration.
- `tokenizer.*`, `chat_template.jinja`, `special_tokens_map.json`: tokenizer assets aligned with the base model.
- `dynahate_instruction_prompt.txt`, `dynahate_system_prompt.txt`: prompt text used for data generation.
- `eval_metrics.txt`, `eval_results.json`, `all_results.json`, `train_results.json`: evaluation + training summaries.

Checkpoints produced during training (e.g., `checkpoint-1000`) are omitted from the upload to keep the repo lightweight.

## Responsible AI & Limitations

- The Dynahate dataset contains explicit hate speech. Applications should include content warnings and guardrails to avoid resurfacing toxic language to end-users.
- The adapter is trained purely as a binary classifier; it does **not** provide rationales or severity levels.
- Llama 3.1’s usage terms apply. Make sure you have the correct license and comply with local regulations when deploying hate-speech classifiers.

## Citation

If you use this work, please cite both the Dynahate dataset and Meta Llama 3.1:

```
@inproceedings{vidgen2020dynahate,
  title={Learning from the Worst: Dynamically Generated Datasets to Improve Online Hate Detection},
  author={Vidgen, Bertie and others},
  booktitle={ACL},
  year={2020}
}
```

```
@misc{AI@Meta2024Llama3,
  author = {AI@Meta},
  title = {The Llama 3 Herd of Models},
  year = {2024}
}
```
