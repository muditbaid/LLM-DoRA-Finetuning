"""
Utilities to score label strings using fine-tuned adapters.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .constants import DEFAULT_BASE_MODEL


def build_llama3_prompt(system: str, instruction: str, user_input: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        + (system or "")
        + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        + (instruction or "")
        + ("\n" + user_input if user_input else "")
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


class LabelScorer:
    def __init__(
        self,
        adapter_path: str,
        label_texts: Sequence[str],
        base_model: str = DEFAULT_BASE_MODEL,
        device: str | None = None,
        ensure_leading_space: bool = True,
        cutoff_len: int = 2048,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cutoff_len = cutoff_len
        self.ensure_leading_space = ensure_leading_space
        self.label_texts = list(label_texts)

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token or "[PAD]"
        self.tokenizer.padding_side = "left"

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.to(self.device)
        self.model.eval()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.label_token_ids: List[torch.Tensor] = []
        for lbl in self.label_texts:
            text = (" " + lbl) if ensure_leading_space and not lbl.startswith(" ") else lbl
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                raise ValueError(f"Label '{lbl}' tokenized to empty ids.")
            self.label_token_ids.append(torch.tensor([ids], device=self.device, dtype=torch.long))

    @torch.inference_mode()
    def score_prompt(self, prompt: str) -> np.ndarray:
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.cutoff_len,
            add_special_tokens=False,
        )
        prompt_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)

        out = self.model(input_ids=prompt_ids, attention_mask=attn, use_cache=True)
        past = out.past_key_values

        log_scores: List[float] = []

        for lbl_ids in self.label_token_ids:
            running_past = past
            prev_tok = lbl_ids[:, :1]

            o = self.model(input_ids=prev_tok, past_key_values=running_past, use_cache=True)
            running_past = o.past_key_values
            logp = F.log_softmax(o.logits[:, -1, :], dim=-1)
            log_scores.append(float(logp[0, prev_tok[0, -1]].item()))

            for t in range(1, lbl_ids.size(1)):
                nxt = lbl_ids[:, t : t + 1]
                o = self.model(input_ids=nxt, past_key_values=running_past, use_cache=True)
                running_past = o.past_key_values
                logp = F.log_softmax(o.logits[:, -1, :], dim=-1)
                log_scores[-1] += float(logp[0, nxt[0, -1]].item())

        return np.array(log_scores, dtype=np.float32)

    @torch.inference_mode()
    def score_prompts(self, prompts: Iterable[str], progress_every: int | None = None) -> np.ndarray:
        scores: List[np.ndarray] = []
        for idx, prompt in enumerate(prompts, start=1):
            scores.append(self.score_prompt(prompt))
            if progress_every and idx % progress_every == 0:
                print(f"[scorer] Processed {idx} prompts…")
        return np.stack(scores, axis=0)

    def close(self) -> None:
        try:
            self.model.to("cpu")
        except Exception:
            pass
        del self.model
