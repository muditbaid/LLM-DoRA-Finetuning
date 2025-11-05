---
language:
- en
license: llama3
tags:
- text-classification
- hate-speech
- qlora
- peft
- llama3
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
pipeline_tag: text-classification
---
# llama3.1-8b-Instruct-qlora-hatexplain

QLoRA adapter trained on the HateXplain dataset using Meta-Llama-3.1-8B-Instruct as the base model. The adapter is provided together with the training recipe and a lightweight inference script.

## Contents

- `adapter/` – QLoRA checkpoint folder exported from LLaMA-Factory (`adapter_model.safetensors`, tokenizer files, training metrics).
- `config/llama31_hatexplain_qlora_sft.yaml` – Training configuration used for supervised fine-tuning.
- `scripts/qlora_inference.py` – Minimal inference helper that loads the base model, merges the adapter weights, and runs generation.

## Usage

1. Install dependencies (PyTorch, `transformers>=4.42`, `peft`, `bitsandbytes`, `accelerate`).
2. Load the model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
adapter_path = "muditbaid/llama3.1-8b-Instruct-qlora-hatexplain"

model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
```

3. Alternatively, run the CLI helper:

```bash
python scripts/qlora_inference.py \
  --adapter-path muditbaid/llama3.1-8b-Instruct-qlora-hatexplain \
  --system-prompt "Classify the text as hate or not." \
  --user-input "Example post here."
```

## Training Notes

- LLaMA-Factory SFT stage with LoRA rank 8, alpha 16, dropout 0.05.
- Cutoff length 2048, cosine scheduler, 3 epochs, learning rate 5e-5.
- QLoRA (4-bit) backbone for efficient fine-tuning on a single GPU.

Refer to `config/llama31_hatexplain_qlora_sft.yaml` for the full set of hyperparameters.
