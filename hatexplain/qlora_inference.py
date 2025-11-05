#!/usr/bin/env python3
"""
Run inference with a base model and a saved QLoRA adapter.

Example:
    python qlora_inference.py \
        --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --adapter-path saves/llama31-8b/kaggle_cyberbullying/qlora \
        --system-prompt "You are a helpful assistant." \
        --user-input "How should we moderate this post?"
"""

import argparse
from typing import Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text with a base model and LoRA/QLoRA adapter."
    )
    parser.add_argument(
        "--base-model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Base model name or local path (default: %(default)s).",
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Path to the trained adapter directory (containing adapter_model.safetensors).",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
        help="System prompt to steer the assistant (set empty string to skip).",
    )
    parser.add_argument(
        "--user-input",
        required=True,
        help="User instruction or text the model should respond to.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature; set <= 0 for deterministic decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p value for nucleus sampling (ignored if temperature <= 0).",
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Disable 4-bit loading; use this if you have a full-precision base model locally.",
    )
    return parser.parse_args()


def build_prompt(
    tokenizer: AutoTokenizer, system_prompt: str, user_input: str
) -> str:
    """Create a conversation prompt using chat templates when available."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_input})

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # Fallback for tokenizers without chat templates.
    if system_prompt:
        return f"{system_prompt}\n\nUser: {user_input}\nAssistant:"
    return f"User: {user_input}\nAssistant:"


def load_model_and_tokenizer(
    base_model: str, adapter_path: str, use_quantization: bool
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the base model, adapter, and tokenizer with sensible defaults."""
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    quantization_config: Optional[BitsAndBytesConfig] = None
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        quantization_config=quantization_config,
    )

    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    return model, tokenizer


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """You are a helpful Assistant."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }

    if temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        generation_kwargs.update({"do_sample": False})

    with torch.inference_mode():
        outputs = model.generate(**generation_kwargs)

    generated_ids = outputs[0, input_ids.shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()

    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        use_quantization=not args.no_quantization,
    )

    prompt = build_prompt(tokenizer, args.system_prompt, args.user_input)
    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("\n=== Model Response ===")
    print(response)


if __name__ == "__main__":
    main()
