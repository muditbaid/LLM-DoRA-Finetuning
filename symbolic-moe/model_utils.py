"""
Model loading helpers for symbolic-moe experts.
"""
from __future__ import annotations

import importlib.util
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

SYMBOLIC_ROOT = Path(__file__).resolve().parent


def _load_config():
    spec = importlib.util.spec_from_file_location("symbolic_moe_model_config", SYMBOLIC_ROOT / "config.py")
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load config module")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules["symbolic_moe_model_config"] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


config_mod = _load_config()
BASE_MODEL = config_mod.BASE_MODEL
ExpertConfig = config_mod.ExpertConfig


@dataclass
class SharedModelRuntime:
    use_quantization: bool = True

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            use_fast=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        q_config: Optional[BitsAndBytesConfig] = None
        if self.use_quantization and torch.cuda.is_available():
            q_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
            quantization_config=q_config,
        )
        self.model.eval()
        self.loaded_adapters: set[str] = set()
        self.active_adapter: Optional[str] = None

    def ensure_adapter(self, config: ExpertConfig) -> None:
        if config.name in self.loaded_adapters:
            return

        adapter_path = str(config.adapter_path)
        if isinstance(self.model, PeftModel):
            self.model.load_adapter(adapter_path, adapter_name=config.name, is_trainable=False)
        else:
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                adapter_name=config.name,
                is_trainable=False,
            )
        self.model.eval()

        self.loaded_adapters.add(config.name)

    def activate_adapter(self, adapter_name: str) -> None:
        if adapter_name not in self.loaded_adapters:
            raise ValueError(f"Adapter '{adapter_name}' is not loaded.")
        if isinstance(self.model, PeftModel):
            self.model.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    def base_inference_context(self):
        if isinstance(self.model, PeftModel):
            return self.model.disable_adapter()
        return nullcontext()

    def close(self) -> None:
        try:
            del self.model
        except AttributeError:
            pass
        torch.cuda.empty_cache()


@dataclass
class ExpertModel:
    config: ExpertConfig
    use_quantization: bool = True
    runtime: Optional[SharedModelRuntime] = None

    def __post_init__(self) -> None:
        self._owns_runtime = False
        if self.runtime is not None:
            self.runtime.ensure_adapter(self.config)
            self.runtime.activate_adapter(self.config.name)
            self.tokenizer = self.runtime.tokenizer
            self.model = self.runtime.model
            return

        runtime = SharedModelRuntime(use_quantization=self.use_quantization)
        runtime.ensure_adapter(self.config)
        runtime.activate_adapter(self.config.name)
        self.runtime = runtime
        self._owns_runtime = True
        self.tokenizer = runtime.tokenizer
        self.model = runtime.model

    def _activate(self) -> None:
        if self.runtime is not None:
            self.runtime.activate_adapter(self.config.name)

    def build_prompt(self, system: str, instruction: str, user_input: str) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        user_content = instruction.strip()
        if user_input:
            # Keep instruction and input on separate lines for clarity
            if user_content:
                user_content = f"{user_content}\n{user_input}"
            else:
                user_content = user_input
        messages.append({"role": "user", "content": user_content})

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        prompt = ""
        if system:
            prompt += system + "\n\n"
        prompt += f"User: {user_content}\nAssistant:"
        return prompt

    def predict(self, prompt: str) -> str:
        self._activate()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        with torch.inference_mode():
            output_ids = self.model.generate(**gen_kwargs)

        generated = output_ids[0, input_ids.shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def predict_with_confidence(self, prompt: str) -> tuple[str, float]:
        """Greedy label with log-prob scoring over configured label_texts."""
        self._activate()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        label_token_ids = [
            self.tokenizer.encode(label, add_special_tokens=False) for label in self.config.label_texts
        ]

        scores = []
        with torch.inference_mode():
            for tokens in label_token_ids:
                label_ids = torch.tensor([tokens], device=self.model.device, dtype=torch.long)
                ids = torch.cat([input_ids, label_ids], dim=1)
                mask = torch.cat([attention_mask, torch.ones_like(label_ids)], dim=1) if attention_mask is not None else None
                outputs = self.model(input_ids=ids, attention_mask=mask)
                log_probs = torch.log_softmax(outputs.logits, dim=-1)
                prompt_len = input_ids.size(1)
                score = 0.0
                for idx, tok in enumerate(tokens):
                    pos = prompt_len - 1 + idx
                    score += float(log_probs[0, pos, tok].item())
                scores.append(score)

        # pick best label, compute softmax confidence over scores
        import math

        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        probs = [s / total for s in exp_scores]

        best_idx = scores.index(max_score)
        best_label = self.config.label_texts[best_idx]
        best_prob = probs[best_idx] if probs else 0.0
        return best_label, best_prob

    def close(self) -> None:
        if self.runtime is not None and not self._owns_runtime:
            return
        if self.runtime is not None and self._owns_runtime:
            self.runtime.close()
            return
        try:
            del self.model
        except AttributeError:
            pass
        torch.cuda.empty_cache()
