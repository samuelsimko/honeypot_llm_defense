# honeypot_simple.py

import os
import json
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, PreTrainedTokenizer
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, field
import argparse
from functools import partial
import random

# -----------------------------
# 1. Hyperparameters (defaults; can be overridden via CLI)
# -----------------------------
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 1
DEFAULT_LR = 2e-4
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Loss weights (benign & honeypot encourage; harmful discourages)
DEFAULT_W_BENIGN = 1.0
DEFAULT_W_HONEYPOT = 1.0
DEFAULT_W_HARMFUL = -0.5  # negative -> gradient ascent on harmful outputs

# -----------------------------
# 2. Tokenizer helpers
# -----------------------------
def tokenize_chat_generic(
    prompts: List[str],
    responses: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256
) -> List[Dict[str, torch.Tensor]]:
    """Standard causal labeling: only assistant tokens get labels; user tokens masked to -100."""
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must match length")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out = []
    for prompt, response in zip(prompts, responses):
        # Full convo (labels will be on assistant span only)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        full_tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )  # (1, L)

        # Prompt-only to find split point
        prompt_only = [{"role": "user", "content": prompt}]
        prompt_tokens = tokenizer.apply_chat_template(
            prompt_only,
            tokenize=True,
            add_generation_prompt=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )  # (1, P)

        labels = full_tokens.clone()
        prompt_len = prompt_tokens.shape[-1]
        labels[0, :prompt_len] = -100  # mask user + template

        L = full_tokens.shape[-1]
        pad_len = max_length - L

        input_ids = torch.nn.functional.pad(full_tokens, (0, pad_len), value=tokenizer.pad_token_id).squeeze(0)
        labels = torch.nn.functional.pad(labels, (0, pad_len), value=-100).squeeze(0)
        attn = torch.ones_like(full_tokens)
        attn = torch.nn.functional.pad(attn, (0, pad_len), value=0).squeeze(0)

        out.append({
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels
        })
    return out

class ChatDataset(Dataset):
    def __init__(self, data: List[Dict[str, torch.Tensor]]):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# -----------------------------
# 3. Data loading
# -----------------------------
def load_ultrachat(num_samples: int, seed: int = 42) -> Tuple[List[str], List[str]]:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=seed).select(range(num_samples))
    prompts, responses = [], []
    for item in ds:
        m = item["messages"]
        if not m: continue
        user = next((x["content"] for x in m if x["role"] == "user"), "")
        # take the first assistant reply if present
        asst = next((x["content"] for x in m if x["role"] == "assistant"), "")
        prompts.append(user)
        responses.append(asst)
    return prompts, responses

def load_circuit_breakers_harmful(path: str, limit: Optional[int]) -> Tuple[List[str], List[str]]:
    """Loads CB JSON (array) or JSONL; returns (prompts, harmful_outputs)."""
    p = Path(path)
    recs: List[Dict[str, Any]] = []
    if p.suffix.lower() == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                recs.append(json.loads(line))
    else:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list): recs = data
            else: raise ValueError("CB file must be list of objects")

    if limit: recs = recs[:limit]
    prompts = [r.get("prompt", "") for r in recs]
    harmful = [r.get("output", "") for r in recs]  # CB 'output' is the harmful target
    return prompts, harmful

def load_honeypots(path: str, limit: Optional[int]) -> Tuple[List[str], List[str]]:
    """Load honeypot dataset: each record has 'prompt' and 'response'.
    Supports both JSON array and JSONL (one object per line) formats.
    """
    recs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        # ✅ JSONL if first non-space isn't '['
        if first_char.strip() != "[":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping bad line: {e}")
        else:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON file must be a list of objects.")
            recs = data

    if limit:
        recs = recs[:limit]

    prompts = [r.get("prompt", "") for r in recs]
    hp = [r.get("response", "") for r in recs]
    return prompts, hp


# -----------------------------
# 4. Tri-batch loader & trainer
# -----------------------------
class TriBatchLoader:
    def __init__(self, benign_loader, harmful_loader, honeypot_loader):
        self.benign_loader = benign_loader
        self.harmful_loader = harmful_loader
        self.honeypot_loader = honeypot_loader
        self._len = min(len(benign_loader), len(harmful_loader), len(honeypot_loader))
    def __iter__(self):
        return zip(self.benign_loader, self.harmful_loader, self.honeypot_loader)
    def __len__(self): return self._len

def ce_loss(model, batch):
    out = model(**{k: v.to(model.device) for k, v in batch.items()})
    logits = out.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = batch["labels"][..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )
    return loss

class TriBatchTrainer(Trainer):
    def __init__(self, benign_ds, harmful_ds, honeypot_ds,
                 w_benign: float, w_harmful: float, w_honeypot: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benign_ds = benign_ds
        self.harmful_ds = harmful_ds
        self.honeypot_ds = honeypot_ds
        self.w_benign = w_benign
        self.w_harmful = w_harmful
        self.w_honeypot = w_honeypot

        # moving averages for nicer logs
        self.ma_benign = None
        self.ma_harmful = None
        self.ma_honeypot = None

    def get_train_dataloader(self):
        args = self.args
        benign_loader = DataLoader(self.benign_ds, batch_size=args.per_device_train_batch_size, shuffle=True)
        harmful_loader = DataLoader(self.harmful_ds, batch_size=args.per_device_train_batch_size, shuffle=True)
        honeypot_loader = DataLoader(self.honeypot_ds, batch_size=args.per_device_train_batch_size, shuffle=True)
        return TriBatchLoader(benign_loader, harmful_loader, honeypot_loader)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=False):
        batch_benign, batch_harmful, batch_honeypot = inputs

        loss_benign = ce_loss(model, batch_benign)
        loss_harmful_ce = ce_loss(model, batch_harmful)   # positive CE number (we'll apply negative weight)
        loss_honeypot = ce_loss(model, batch_honeypot)

        # total: encourage benign & honeypot, discourage harmful
        total_loss = (
            self.w_benign * loss_benign +
            self.w_honeypot * loss_honeypot +
            self.w_harmful * loss_harmful_ce
        )

        # moving averages for logging
        def ema(prev, val, a=0.9): return val.item() if prev is None else a*prev + (1-a)*val.item()
        self.ma_benign = ema(self.ma_benign, loss_benign)
        self.ma_harmful = ema(self.ma_harmful, loss_harmful_ce)
        self.ma_honeypot = ema(self.ma_honeypot, loss_honeypot)

        # log per-step (Trainer will throttle per logging_steps)
        self.log({
            "loss/benign_ce": loss_benign.item(),
            "loss/honeypot_ce": loss_honeypot.item(),
            "loss/harmful_ce": loss_harmful_ce.item(),
            "ema/benign_ce": self.ma_benign,
            "ema/honeypot_ce": self.ma_honeypot,
            "ema/harmful_ce": self.ma_harmful,
            "loss/total": total_loss.item(),
        })

        return (total_loss, None) if return_outputs else total_loss

# -----------------------------
# 5. Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--ultrachat_samples", type=int, default=5000)
    parser.add_argument("--cb_path", type=str, required=True, help="Path to circuit_breakers_train.json/jsonl (with harmful 'output')")
    parser.add_argument("--honeypot_path", type=str, required=True, help="Path to honeypots json/jsonl (with 'prompt' and 'response')")
    parser.add_argument("--limit_cb", type=int, default=5000, help="Limit CB examples")
    parser.add_argument("--limit_hp", type=int, default=5000, help="Limit honeypot examples")
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--w_benign", type=float, default=DEFAULT_W_BENIGN)
    parser.add_argument("--w_harmful", type=float, default=DEFAULT_W_HARMFUL)
    parser.add_argument("--w_honeypot", type=float, default=DEFAULT_W_HONEYPOT)
    parser.add_argument("--output_dir", type=str, default="./honeypot_simple")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--grad_accum", type=int, default=2)
    args = parser.parse_args()

    # 1) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) Load datasets
    # Benign (UltraChat)
    safe_prompts, safe_responses = load_ultrachat(args.ultrachat_samples)
    safe_data = tokenize_chat_generic(safe_prompts, safe_responses, tokenizer, max_length=args.max_length)

    # Harmful (CB 'output' is harmful content we want to DECREASE)
    harmful_prompts, harmful_outputs = load_circuit_breakers_harmful(args.cb_path, args.limit_cb)
    harmful_data = tokenize_chat_generic(harmful_prompts, harmful_outputs, tokenizer, max_length=args.max_length)

    # Honeypots (your generated "response" we want to INCREASE)
    hp_prompts, hp_outputs = load_honeypots(args.honeypot_path, args.limit_hp)
    # Make sure prompts align semantically with harmful prompts; if sizes differ, zip via dataloader min-len
    honeypot_data = tokenize_chat_generic(hp_prompts, hp_outputs, tokenizer, max_length=args.max_length)

    safe_ds = ChatDataset(safe_data)
    harmful_ds = ChatDataset(harmful_data)
    honeypot_ds = ChatDataset(honeypot_data)

    # 3) Model + LoRA
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.to(args.device)
    model.train()

    # 4) Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_grad_norm=1.0,
        save_total_limit=args.save_total_limit,
        report_to="none"
    )

    # 5) Trainer
    trainer = TriBatchTrainer(
        model=model,
        benign_ds=safe_ds,
        harmful_ds=harmful_ds,
        honeypot_ds=honeypot_ds,
        w_benign=args.w_benign,
        w_harmful=args.w_harmful,
        w_honeypot=args.w_honeypot,
        args=training_args,
        tokenizer=tokenizer
    )

    # 6) Train
    trainer.train()

    # 7) Save LoRA adapter
    model.save_pretrained(os.path.join(args.output_dir, "lora_adapter"))

if __name__ == "__main__":
    main()
