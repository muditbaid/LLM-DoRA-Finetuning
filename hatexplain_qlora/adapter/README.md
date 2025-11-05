---
library_name: peft
license: other
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
tags:
- base_model:adapter:meta-llama/Meta-Llama-3.1-8B-Instruct
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: qlora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qlora

This model is a fine-tuned version of [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) on the hatexplain_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2349

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 3e-05
- train_batch_size: 3
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 24
- optimizer: Use adamw_torch_fused with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.05
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.225         | 0.7800 | 500  | 0.2405          |
| 0.2144        | 1.5601 | 1000 | 0.2299          |
| 0.195         | 2.3401 | 1500 | 0.2424          |


### Framework versions

- PEFT 0.17.1
- Transformers 4.57.1
- Pytorch 2.9.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.1