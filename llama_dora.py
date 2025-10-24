# nohup python llama_dora.py > hx_train.log 2>&1 &
# ps -ef | grep llama_dora.py
# tail -f hx_train.log

import os
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from datasets import Dataset
from sklearn.metrics import classification_report

# pip install -U "transformers>=4.43" "peft>=0.11.0" accelerate datasets sentencepiece
# Optional: pip install flash-attn --no-build-isolation  (if your CUDA setup supports it)


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import transformers


# Prompt template kept separate so it can be reused for eval/inference.
def build_prompt(post: str) -> str:
    """Return the instruction prompt for a single social media post."""
    post = (post or "").strip()
    return (
        "You are a language model trained to evaluate social media posts and determine "
        "whether the content constitutes Hate Speech or Not Hate Speech based on a provided definition.\n\n"
        "Definition of Hate Speech:\n"
        "Hate speech refers to any language that attacks, diminishes, incites violence against, "
        "or promotes hatred toward individuals or groups based on characteristics such as physical appearance, "
        "religion, descent, national or ethnic origin, sexual orientation, gender identity, or other inherent attributes. "
        "Hate speech may be conveyed overtly or subtly, and can include sarcastic, humorous, or coded expressions "
        "intended to convey hostility or exclusion.\n\n"
        f"Post: {post}\n"
        "Your task is to analyze the post and then return a JSON object:\n"
        "{\n"
        '  "label": "Hate" | "Not Hate"\n'
        "}"
    )

if not hasattr(transformers, "EncoderDecoderCache"):
    # Provide a minimal shim so newer PEFT releases keep working on slightly older transformers builds.
    from transformers.cache_utils import Cache, DynamicCache

    class _EncoderDecoderCacheShim(Cache):
        def __init__(self, encoder_cache=None, self_attention_cache=None, cross_attention_cache=None):
            self.encoder_cache = encoder_cache or DynamicCache()
            self.self_attention_cache = self_attention_cache or DynamicCache()
            self.cross_attention_cache = cross_attention_cache or DynamicCache()

        @property
        def key_cache(self):
            return self.self_attention_cache.key_cache

        @key_cache.setter
        def key_cache(self, value):
            self.self_attention_cache.key_cache = value

        @property
        def value_cache(self):
            return self.self_attention_cache.value_cache

        @value_cache.setter
        def value_cache(self, value):
            self.self_attention_cache.value_cache = value

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            return self.self_attention_cache.update(key_states, value_states, layer_idx, cache_kwargs)

        def get_seq_length(self, layer_idx=None):
            return self.self_attention_cache.get_seq_length(layer_idx)

        def get_max_length(self):
            return self.self_attention_cache.get_max_length()

        def get_usable_length(self, new_seq_length, layer_idx=None):
            return self.self_attention_cache.get_usable_length(new_seq_length, layer_idx)

        def reorder_cache(self, beam_idx):
            self.self_attention_cache.reorder_cache(beam_idx)
            if hasattr(self.cross_attention_cache, "reorder_cache"):
                self.cross_attention_cache.reorder_cache(beam_idx)

        def to_legacy_cache(self):
            decoder = (
                self.self_attention_cache.to_legacy_cache()
                if hasattr(self.self_attention_cache, "to_legacy_cache")
                else None
            )
            encoder = (
                self.encoder_cache.to_legacy_cache()
                if hasattr(self.encoder_cache, "to_legacy_cache")
                else None
            )
            cross = (
                self.cross_attention_cache.to_legacy_cache()
                if hasattr(self.cross_attention_cache, "to_legacy_cache")
                else None
            )

            parts = [part for part in (encoder, decoder, cross) if part is not None]
            if not parts:
                return None
            if len(parts) == 1:
                return parts[0]
            return tuple(parts)

        @classmethod
        def from_legacy_cache(cls, past_key_values=None):
            if past_key_values is None:
                return cls()

            encoder = None
            decoder = past_key_values
            cross = None

            if isinstance(past_key_values, tuple):
                if len(past_key_values) == 3:
                    encoder, decoder, cross = past_key_values
                elif len(past_key_values) == 2:
                    encoder, decoder = past_key_values

            def _ensure_cache(cache_data):
                if cache_data is None:
                    return DynamicCache()
                if isinstance(cache_data, DynamicCache):
                    return cache_data
                return DynamicCache.from_legacy_cache(cache_data)

            encoder_cache = _ensure_cache(encoder)
            decoder_cache = _ensure_cache(decoder)
            cross_cache = _ensure_cache(cross)

            cache = cls(encoder_cache=encoder_cache, self_attention_cache=decoder_cache, cross_attention_cache=cross_cache)
            cache.is_updated = {}
            return cache

    transformers.EncoderDecoderCache = _EncoderDecoderCacheShim

MAX_SEQ_LENGTH = 1024
# Toggle to control evaluation during training; enable when metrics are needed.
ENABLE_EVAL_DURING_TRAINING = True

def preprocess_dataset(
    df,
    text_col,
    label_col,
    tokenizer,
    max_len=MAX_SEQ_LENGTH,
    split_ratio=0.1,
    seed=42,
    max_samples=None,
):
    """
    Prepare a dataset for instruction-tuning with text labels.
    Each row becomes: <prompt> + <label_text> for causal LM training.
    
    Args:
        df (pd.DataFrame): input dataframe
        text_col (str): column name containing the input text
        label_col (str): column name containing the target label (string, e.g. "Hate")
        tokenizer: Hugging Face tokenizer
        max_len (int): max sequence length
        split_ratio (float): fraction to use for eval set (default 0.1)
        seed (int): random seed for reproducibility
        max_samples (int|None): optional cap on total rows before splitting
    """
    ds = Dataset.from_pandas(df)

    if max_samples and max_samples > 0:
        take = min(int(max_samples), len(ds))
        ds = ds.shuffle(seed=seed).select(range(take))

    def format_example(ex):
        # Build an instruction-style prompt
        prompt = build_prompt(ex[text_col])
        raw_label = str(ex[label_col]).strip()
        label_text = "Hate" if raw_label.lower() == "hate" else "Not Hate"
        target_json = '{\n  "label": "' + label_text + '"\n}'

        # Full sequence = prompt + ground truth JSON label
        full_text = prompt + "\n" + target_json

        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_len,
            padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = ds.map(format_example, batched=False, remove_columns=ds.column_names)
    split_ds = tokenized.train_test_split(test_size=split_ratio, seed=seed)
    return split_ds["train"], split_ds["test"]


def main():
    # Heavy imports and model setup live inside main to avoid import-time side effects
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    import torch, bitsandbytes as bnb
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
    print("CUDA available:", torch.cuda.is_available())
    print("Current device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    # bitsandbytes exposes utility helpers in different ways across versions.
    # Guard the check so older/newer bnb releases don't raise AttributeError.
    try:
        is_cublas_fn = getattr(bnb.utils, "is_cublasLt_available", None)
        if callable(is_cublas_fn):
            try:
                cublaslt_available = is_cublas_fn()
            except Exception as _e:
                cublaslt_available = f"error calling is_cublasLt_available(): {_e}"
        else:
            cublaslt_available = "unknown (bitsandbytes.utils.is_cublasLt_available not present)"
    except Exception as _e:
        # In case bnb.utils itself is missing or something else goes wrong
        cublaslt_available = f"error checking cublasLt availability: {_e}"

    print("cublasLt available (bnb):", cublaslt_available)

    bf16_ok = False
    if torch.cuda.is_available():
        is_bf16_fn = getattr(torch.cuda, "is_bf16_supported", None)
        bf16_ok = bool(is_bf16_fn() if callable(is_bf16_fn) else torch.cuda.get_device_capability(0)[0] >= 8)
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    base_model = "meta-llama/Llama-3.1-8B-Instruct"  # or the 8B base if you prefer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    # Llama tokenizers usually need this set for padding:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load in bf16 to keep memory reasonable (A100/MI300/BF16-capable GPU)
    # Load config first so we can sanitize `rope_scaling` entries that some
    # Llama releases provide with slightly different keys (e.g., `rope_type`).
    # The transformers PretrainedConfig expects `rope_scaling` to contain at
    # least `type` and `factor`.
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        rs = getattr(config, "rope_scaling", None)
        if isinstance(rs, dict):
            # If the dict already matches the expected minimal shape, keep it.
            if "type" in rs and "factor" in rs:
                config.rope_scaling = rs
            else:
                # Unknown/unsupported shape (e.g., rope_type='llama3'). Remove it
                # so transformers does not raise during config validation.
                print("Notice: removing unsupported `rope_scaling` from config to avoid validation errors.")
                config.rope_scaling = None
    except Exception as e:
        print("Warning: could not load/sanitize model config:", e)
        config = None

    # Try to load the model; if the remote repo's config has an unexpected
    # `rope_scaling` shape (common for some Llama3 configs) Transformers
    # validation will raise a ValueError. In that case, download the raw
    # config.json, patch `rope_type` -> `type` and ensure `factor` is present,
    # then load the model using the patched config from a temporary directory.
    # Wrap model loading in a two-step attempt: first try with requested
    # quantization (8-bit via bitsandbytes). If bitsandbytes fails with the
    # 'SCB' AttributeError (compatibility), retry without 8-bit to fall back
    # to a non-bitsandbytes load (may require more GPU memory).
    def _load_with_config(cfg):
        return AutoModelForCausalLM.from_pretrained(
            base_model,
            config=cfg,
            quantization_config=quant_config,
            device_map={"": 0},
            use_safetensors=True,
            trust_remote_code=True,
        )

    try:
        model = _load_with_config(config)
    except ValueError as e:
        err_msg = str(e)
        if "rope_scaling" in err_msg:
            try:
                from huggingface_hub import hf_hub_download
                import json, tempfile

                print("Patching remote config.json to fix rope_scaling format...")
                cfg_path = hf_hub_download(repo_id=base_model, filename="config.json")
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)

                # Remove or neutralize problematic rope_scaling entries which
                # transformers may reject (common in some Llama-3 configs).
                if "rope_scaling" in cfg:
                    rs = cfg.get("rope_scaling")
                    if not (isinstance(rs, dict) and "type" in rs and "factor" in rs):
                        print("Patching out unsupported rope_scaling from downloaded config.json")
                        cfg.pop("rope_scaling", None)

                # write patched config to a temp dir and load config from there
                tmpd = tempfile.mkdtemp(prefix="patched_cfg_")
                patched_cfg_file = os.path.join(tmpd, "config.json")
                with open(patched_cfg_file, "w", encoding="utf-8") as f:
                    json.dump(cfg, f)

                patched_config = AutoConfig.from_pretrained(tmpd, trust_remote_code=True)
                try:
                    model = _load_with_config(patched_config)
                except AttributeError as ae:
                    # Bitsandbytes compatibility issue (SCB attribute missing)
                    if "SCB" in str(ae) or "bitsandbytes" in str(ae):
                        print("Detected bitsandbytes compatibility error while loading patched config. Retrying without 8-bit quantization...")
                        # Retry with quantization disabled
                        quant_config_fallback = BitsAndBytesConfig(load_in_8bit=False)
                        model = AutoModelForCausalLM.from_pretrained(
                            base_model,
                            config=patched_config,
                            quantization_config=quant_config_fallback,
                            device_map="auto",
                            use_safetensors=True,
                            trust_remote_code=True,
                        )
                    else:
                        raise
            except Exception as e2:
                print("Failed to patch remote config:", e2)
                raise
        else:
            raise
    except AttributeError as ae:
        # Catch the bitsandbytes 'SCB' case and retry without 8-bit
        if "SCB" in str(ae) or "bitsandbytes" in str(ae):
            print("bitsandbytes appears incompatible with this PyTorch build (SCB error). Retrying model load with 8-bit disabled.")
            quant_config_fallback = BitsAndBytesConfig(load_in_8bit=False)
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                config=config,
                quantization_config=quant_config_fallback,
                device_map="auto",
                use_safetensors=True,
                trust_remote_code=True,
            )
        else:
            raise

    # If we still hit bitsandbytes runtime errors (e.g., `.to` not supported or
    # SCB attribute missing), try a last-resort load without any
    # quantization_config so that transformers/accelerate won't instantiate
    # bitsandbytes modules. Use low_cpu_mem_usage to avoid extra state_dict
    # operations where possible.
    try:
        # Quick check: if model exists and is loaded, skip this.
        if 'model' in locals() and model is not None:
            pass
        else:
            model
    except Exception:
        # If model is not successfully assigned, attempt the final fallback
        try:
            print("Final fallback: loading model without BitsAndBytes quantization (this will use more memory).")
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                config=config,
                quantization_config=None,
                device_map="auto",
                use_safetensors=True,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        except Exception as final_e:
            print("Final fallback also failed:", final_e)
            raise


    # Prepare model for k-bit + gradient checkpointing before wrapping with PEFT.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "config"):
        model.config.use_cache = False
        if getattr(model.config, "pretraining_tp", None) is None:
            model.config.pretraining_tp = 1

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # DoRA via PEFT: enable with use_dora=True
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        use_dora=True,
        # Llama 3.x target modules:
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # sanity check

    hx = pd.read_csv('../Hate Speech/hatexplain_processed.csv')
    dataset_size = len(hx)
    print(f"Loaded dataset with {dataset_size} rows.")

    # Dataset window
    SAMPLE_START = 0
    SAMPLE_END = 100  # exclusive; set to None to run to end
    SAMPLE_LIMIT = None

    if SAMPLE_START or SAMPLE_END:
        start = max(SAMPLE_START or 0, 0)
        stop = SAMPLE_END if SAMPLE_END is not None else dataset_size
        stop = min(stop, dataset_size)
        if start >= stop:
            raise ValueError(f"Invalid slice [{start}:{stop}] for dataset size {dataset_size}.")
        hx = hx.iloc[start:stop].reset_index(drop=True)
        print(f"Sliced dataset rows [{start}:{stop}) -> {len(hx)} rows.")

    if SAMPLE_LIMIT and SAMPLE_LIMIT > 0:
        capped = min(int(SAMPLE_LIMIT), len(hx))
        print(f"Limiting dataset to {capped} samples (SAMPLE_LIMIT={SAMPLE_LIMIT}).")
        hx = hx.sample(n=capped, random_state=42).reset_index(drop=True)
    else:
        SAMPLE_LIMIT = None

    # Preprocess dataset
    train_ds, eval_ds = preprocess_dataset(
        hx,
        text_col="text",
        label_col="class_label",   # e.g., contains "Hate" or "Not Hate"
        tokenizer=tokenizer,
        max_len=MAX_SEQ_LENGTH,
        max_samples=SAMPLE_LIMIT,
    )
    print(train_ds[0])

    # Metrics and trainer
    label_pattern = re.compile(r'"label"\s*:\s*"(Hate|Not Hate)"', re.IGNORECASE)

    def _extract_label(text: str) -> int:
        matches = label_pattern.findall(text)
        if not matches:
            return -1
        label = matches[-1].strip().lower()
        if label == "hate":
            return 1
        if label == "not hate":
            return 0
        return -1

    metric_state = {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "total": 0,
    }

    def compute_metrics(eval_pred, compute_result=False):
        predictions = getattr(eval_pred, "predictions", None)
        labels = getattr(eval_pred, "label_ids", None)

        if predictions is None or labels is None:
            predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        if isinstance(predictions, torch.Tensor):
            token_ids = torch.argmax(predictions, dim=-1).detach().cpu()
        else:
            predictions_arr = np.asarray(predictions)
            token_ids = (
                np.argmax(predictions_arr, axis=-1)
                if predictions_arr.ndim >= 2
                else predictions_arr
            )

        def _to_token_lists(batch, *, replace_mask=False):
            if batch is None:
                return []

            if isinstance(batch, torch.Tensor):
                batch = batch.detach().cpu().tolist()
            elif isinstance(batch, np.ndarray):
                batch = batch.tolist()
            elif isinstance(batch, (np.generic,)):
                batch = [int(batch)]
            elif isinstance(batch, tuple):
                batch = list(batch)

            if isinstance(batch, list):
                normalized = []
                for seq in batch:
                    if isinstance(seq, torch.Tensor):
                        seq = seq.detach().cpu().tolist()
                    elif isinstance(seq, np.ndarray):
                        seq = seq.tolist()

                    if isinstance(seq, (list, tuple)):
                        tokens = [int(tok) for tok in seq]
                    else:
                        tokens = [int(seq)]

                    if replace_mask:
                        tokens = [
                            tokenizer.pad_token_id if tok == -100 else int(tok)
                            for tok in tokens
                        ]
                    normalized.append(tokens)
                return normalized

            return [[int(batch)]]

        pred_token_seqs = _to_token_lists(token_ids)

        if isinstance(labels, tuple):
            labels = labels[0]
        label_token_seqs = _to_token_lists(labels, replace_mask=True)

        batch_tp = batch_tn = batch_fp = batch_fn = 0
        batch_total = 0

        for pred_seq, label_seq in zip(pred_token_seqs, label_token_seqs):
            pred_text = tokenizer.decode(pred_seq, skip_special_tokens=True)
            label_text = tokenizer.decode(label_seq, skip_special_tokens=True)

            pred_label = _extract_label(pred_text)
            pred_label = 0 if pred_label == -1 else pred_label

            gold_label = _extract_label(label_text)
            gold_label = 0 if gold_label == -1 else gold_label

            batch_total += 1
            if pred_label == 1 and gold_label == 1:
                batch_tp += 1
            elif pred_label == 0 and gold_label == 0:
                batch_tn += 1
            elif pred_label == 1 and gold_label == 0:
                batch_fp += 1
            elif pred_label == 0 and gold_label == 1:
                batch_fn += 1

        metric_state["tp"] += batch_tp
        metric_state["tn"] += batch_tn
        metric_state["fp"] += batch_fp
        metric_state["fn"] += batch_fn
        metric_state["total"] += batch_total

        if not compute_result:
            return {}

        total = metric_state["total"] or 1
        accuracy = (metric_state["tp"] + metric_state["tn"]) / total

        precision = (
            metric_state["tp"] / (metric_state["tp"] + metric_state["fp"])
            if (metric_state["tp"] + metric_state["fp"])
            else 0.0
        )
        recall = (
            metric_state["tp"] / (metric_state["tp"] + metric_state["fn"])
            if (metric_state["tp"] + metric_state["fn"])
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        result = {"accuracy": accuracy, "f1": f1}

        metric_state.update({"tp": 0, "tn": 0, "fp": 0, "fn": 0, "total": 0})
        return result

    from trl import SFTTrainer, SFTConfig

    available_cpus = os.cpu_count() or 1
    dataloader_workers = min(8, max(1, available_cpus // 2))
    print(f"Detected {available_cpus} CPU cores; dataloader workers set to {dataloader_workers}.")

    sft_args = SFTConfig(
        output_dir="llama31-8b-dora_hx",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        learning_rate=5e-5,
        logging_steps=1,
        fp16=not bf16_ok,
        bf16=bf16_ok,
        eval_strategy="epoch" if ENABLE_EVAL_DURING_TRAINING else "no",
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        eval_do_concat_batches=False,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="tensorboard",
        dataloader_num_workers=dataloader_workers,
        optim="paged_adamw_8bit",
        batch_eval_metrics=ENABLE_EVAL_DURING_TRAINING,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds if ENABLE_EVAL_DURING_TRAINING else None,
        args=sft_args,
        compute_metrics=compute_metrics if ENABLE_EVAL_DURING_TRAINING else None,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(sft_args.output_dir)


if __name__ == '__main__':
    main()
