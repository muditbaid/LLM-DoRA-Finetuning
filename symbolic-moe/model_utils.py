"""
Model loading helpers for symbolic-moe experts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .config import BASE_MODEL, ExpertConfig


@dataclass
class ExpertModel:
    config: ExpertConfig
    use_quantization: bool = True

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.adapter_path,
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
        self.model = PeftModel.from_pretrained(
            self.model,
            str(self.config.adapter_path),
            is_trainable=False,
        )
        self.model.eval()

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

    def close(self) -> None:
        try:
            del self.model
        except AttributeError:
            pass
        torch.cuda.empty_cache()
