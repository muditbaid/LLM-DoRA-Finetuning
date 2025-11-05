
# LLM-DoRA-Finetuning

This directory aggregates the ready-to-share QLoRA artifacts for the HateXplain and Kaggle Cyberbullying experiments. Every task subfolder bundles:

- `adapter/`: final LoRA adapter weights plus tokenizer assets (checkpoint directories removed).
- `*.yaml`: LLaMA-Factory training configuration used for fine-tuning.
- `qlora_train_*.log`: raw training log from `llamafactory-cli`.
- Data prep scripts, prompt text files (when applicable), and evaluation helpers.
- `qlora_inference.py`: minimal inference entrypoint targeting the saved adapter.

## HateXplain Workflow

```bash
# 1. Prepare data (requires cached dataset or HF access)
../llama_factory_env/bin/python LLaMA-Factory/scripts/prepare_hatexplain.py   --output-prefix hatexplain \
  --splits train validation test \
  --prompt-mode both

# 2. Train QLoRA adapter
nohup env CUDA_VISIBLE_DEVICES=0 llamafactory-cli train   examples/train_qlora/llama31_hatexplain_qlora_sft.yaml   > LLaMA-Factory/qlora_train_hatexplain.log 2>&1 &

# 3. Evaluate
../llama_factory_env/bin/python LLaMA-Factory/scripts/eval_hatexplain_metrics.py   --adapter LLM-DoRA-Finetuning/hatexplain/adapter \
  --split validation

# 4. Run inference
python hatexplain/qlora_inference.py   --adapter-path hatexplain/adapter \
  --system-prompt "Classify the following post" \
  --user-input "Example post to classify"
```

## Kaggle Cyberbullying Workflow

```bash
# 1. Prepare data (expects kaggle_cyberbullying.csv in repo root)
../llama_factory_env/bin/python LLaMA-Factory/scripts/generate_kaggle_cyberbullying_jsonl.py   kaggle_cyberbullying.csv LLaMA-Factory/data \
  --prompt-mode both

# 2. Train QLoRA adapter
nohup env CUDA_VISIBLE_DEVICES=0 llamafactory-cli train   examples/train_qlora/llama31_kaggle_cyberbullying_qlora_sft.yaml   > LLaMA-Factory/qlora_train_kaggle.log 2>&1 &

# 3. Evaluate
../llama_factory_env/bin/python LLaMA-Factory/scripts/eval_cyberbullying_metrics.py   --adapter LLM-DoRA-Finetuning/kaggle_cyberbullying/adapter \
  --split validation

# 4. Run inference
python kaggle_cyberbullying/qlora_inference.py   --adapter-path kaggle_cyberbullying/adapter \
  --system-prompt "Classify the post" \
  --user-input "Example tweet"
```

Tweak `CUDA_VISIBLE_DEVICES`, file paths, or prompt strings as needed. Evaluation scripts expose additional flags for alternative splits/metrics; check their docstrings for details.
