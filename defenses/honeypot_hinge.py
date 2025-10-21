import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, PreTrainedTokenizer, TrainerCallback
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import argparse
from contextlib import contextmanager
import math
import csv
import matplotlib.pyplot as plt

# -----------------------------
# 1. Defaults
# -----------------------------
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 1
DEFAULT_LR = 2e-4
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_W_BENIGN = 1.0
DEFAULT_W_REFUSAL = 1.25
DEFAULT_W_HONEYPOT = 1.0
DEFAULT_W_HARMFUL = 0.0
DEFAULT_W_ALIGN_BENIGN = 0.1
DEFAULT_W_ALIGN_REFUSAL = 0.2
DEFAULT_W_ALIGN_HONEYPOT = 0.0

DEFAULT_MARGIN_REF_GT_HP = 0.25
DEFAULT_MARGIN_HP_GT_HARM = 0.25
DEFAULT_W_MARGIN_REF_HP = 0.5
DEFAULT_W_MARGIN_HP_HARM = 0.5
DEFAULT_MARGIN_HARMFUL_CE = 80.0
DEFAULT_W_MARGIN_HARMFUL_CE = 0.0

DEFAULT_REFUSAL_RESPONSE = "I'm sorry, I cannot do that."

# -----------------------------
# 2. Tokenization helpers
# -----------------------------
def tokenize_chat_generic(prompts: List[str], responses: List[str],
                          tokenizer: PreTrainedTokenizer, max_length: int = 256):
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must match length")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out = []
    for prompt, response in zip(prompts, responses):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        full_tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            max_length=max_length, truncation=True, return_tensors="pt"
        )

        prompt_only = [{"role": "user", "content": prompt}]
        prompt_tokens = tokenizer.apply_chat_template(
            prompt_only, tokenize=True, add_generation_prompt=True,
            max_length=max_length, truncation=True, return_tensors="pt"
        )

        labels = full_tokens.clone()
        prompt_len = prompt_tokens.shape[-1]
        labels[0, :prompt_len] = -100

        L = full_tokens.shape[-1]
        pad_len = max_length - L

        input_ids = F.pad(full_tokens, (0, pad_len), value=tokenizer.pad_token_id).squeeze(0)
        labels = F.pad(labels, (0, pad_len), value=-100).squeeze(0)
        attn = torch.ones_like(full_tokens)
        attn = F.pad(attn, (0, pad_len), value=0).squeeze(0)

        out.append({
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels
        })
    return out

class ChatDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# -----------------------------
# 3. Data loading
# -----------------------------
def load_ultrachat(num_samples: int, seed: int = 42):
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=seed).select(range(num_samples))
    prompts, responses = [], []
    for item in ds:
        m = item["messages"]
        if not m: continue
        user = next((x["content"] for x in m if x["role"] == "user"), "")
        asst = next((x["content"] for x in m if x["role"] == "assistant"), "")
        prompts.append(user)
        responses.append(asst)
    return prompts, responses

def _load_json_or_jsonl(path: str):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(8192)
        f.seek(0)
        first_non_ws = next((ch for ch in head if not ch.isspace()), None)

        def parse_jsonl(file_obj):
            out = []
            for line in file_obj:
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("//"): continue
                try:
                    out.append(json.loads(s))
                except json.JSONDecodeError:
                    continue
            return out

        if first_non_ws == "[":
            try:
                data = json.load(f)
                if isinstance(data, list): return data
            except Exception:
                f.seek(0)
                return parse_jsonl(f)
        return parse_jsonl(f)

def load_circuit_breakers_harmful(path: str, limit: Optional[int]):
    recs = _load_json_or_jsonl(path)
    if limit: recs = recs[:limit]
    prompts = [r.get("prompt", "") for r in recs]
    harmful = [r.get("output", "") for r in recs]
    return prompts, harmful

def load_circuit_breakers_refusal(path: str, limit: Optional[int]):
    recs = _load_json_or_jsonl(path)
    if limit: recs = recs[:limit]
    prompts = [r.get("prompt", "") for r in recs]
    refusals = [(r.get("llama3_output") or DEFAULT_REFUSAL_RESPONSE) for r in recs]
    return prompts, refusals

def load_honeypots(path: str, limit: Optional[int]):
    recs = _load_json_or_jsonl(path)
    if limit: recs = recs[:limit]
    prompts = [r.get("prompt", "") for r in recs]
    hp = [r.get("response", "") for r in recs]
    return prompts, hp

# -----------------------------
# 4. Utils
# -----------------------------
def ce_loss_from_outputs(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1), ignore_index=-100)

def mean_logprob_per_example(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    mask = (shift_labels != -100).float()
    logp = F.log_softmax(shift_logits, dim=-1)
    gold_logp = logp.gather(-1, shift_labels.unsqueeze(-1).clamp_min(0)).squeeze(-1) * mask
    tok_counts = mask.sum(dim=-1).clamp_min(1.0)
    return gold_logp.sum(dim=-1) / tok_counts

@contextmanager
def adapters_disabled_ctx(model):
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter(): yield
    else: yield

def rep_alignment_loss(model, batch, layer_ids):
    device = model.device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    out_adapt = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    hs_adapt = out_adapt.hidden_states
    with adapters_disabled_ctx(model):
        with torch.no_grad():
            out_base = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
            hs_base = out_base.hidden_states
    num_layers = len(hs_adapt) - 1
    chosen = [max(0, min(num_layers - 1, int(lid))) + 1 for lid in layer_ids]
    mask = (labels != -100).unsqueeze(-1).float()
    losses = []
    for idx in chosen:
        A, B = hs_adapt[idx], hs_base[idx]
        diff = (A - B) * mask
        denom = mask.sum() * A.size(-1) + 1e-8
        losses.append((diff.pow(2).sum()) / denom)
    return sum(losses) / max(1, len(losses))

def align_honeypot(model, batch_benign, batch_honeypot, layer_ids):
    """
    Align honeypot and benign representations to be the same as the original network (L2 distance).
    Also includes margin term for harmful CE loss to keep it under threshold.
    """
    device = model.device
    
    # Get representations for benign data
    benign_input_ids = batch_benign["input_ids"].to(device)
    benign_attention_mask = batch_benign["attention_mask"].to(device)
    benign_labels = batch_benign["labels"].to(device)
    
    out_benign_adapt = model(input_ids=benign_input_ids, attention_mask=benign_attention_mask, 
                           output_hidden_states=True, use_cache=False)
    hs_benign_adapt = out_benign_adapt.hidden_states
    
    # Get representations for honeypot data
    hp_input_ids = batch_honeypot["input_ids"].to(device)
    hp_attention_mask = batch_honeypot["attention_mask"].to(device)
    hp_labels = batch_honeypot["labels"].to(device)
    
    out_hp_adapt = model(input_ids=hp_input_ids, attention_mask=hp_attention_mask, 
                        output_hidden_states=True, use_cache=False)
    hs_hp_adapt = out_hp_adapt.hidden_states
    
    # Get original network representations (without adapters)
    with adapters_disabled_ctx(model):
        with torch.no_grad():
            out_benign_base = model(input_ids=benign_input_ids, attention_mask=benign_attention_mask, 
                                  output_hidden_states=True, use_cache=False)
            hs_benign_base = out_benign_base.hidden_states
            
            out_hp_base = model(input_ids=hp_input_ids, attention_mask=hp_attention_mask, 
                              output_hidden_states=True, use_cache=False)
            hs_hp_base = out_hp_base.hidden_states
    
    num_layers = len(hs_benign_adapt) - 1
    chosen = [max(0, min(num_layers - 1, int(lid))) + 1 for lid in layer_ids]
    
    # Create masks for valid tokens
    benign_mask = (benign_labels != -100).unsqueeze(-1).float()
    hp_mask = (hp_labels != -100).unsqueeze(-1).float()
    
    alignment_losses = []
    for idx in chosen:
        # Align benign representations to original
        benign_adapt = hs_benign_adapt[idx]
        benign_base = hs_benign_base[idx]
        benign_diff = (benign_adapt - benign_base) * benign_mask
        benign_denom = benign_mask.sum() * benign_adapt.size(-1) + 1e-8
        benign_loss = (benign_diff.pow(2).sum()) / benign_denom
        
        # Align honeypot representations to original
        hp_adapt = hs_hp_adapt[idx]
        hp_base = hs_hp_base[idx]
        hp_diff = (hp_adapt - hp_base) * hp_mask
        hp_denom = hp_mask.sum() * hp_adapt.size(-1) + 1e-8
        hp_loss = (hp_diff.pow(2).sum()) / hp_denom
        
        alignment_losses.append(benign_loss + hp_loss)
    
    return sum(alignment_losses) / max(1, len(alignment_losses))

# -----------------------------
# 5. Quad loader & trainer
# -----------------------------
class QuadBatchLoader:
    def __init__(self, benign_loader, refusal_loader, harmful_loader, honeypot_loader):
        self.benign_loader = benign_loader
        self.refusal_loader = refusal_loader
        self.harmful_loader = harmful_loader
        self.honeypot_loader = honeypot_loader
        self._len = min(len(benign_loader), len(refusal_loader), len(harmful_loader), len(honeypot_loader))
    def __iter__(self):
        return zip(self.benign_loader, self.refusal_loader, self.harmful_loader, self.honeypot_loader)
    def __len__(self): return self._len

class QuadBatchTrainer(Trainer):
    def __init__(self, benign_ds, refusal_ds, harmful_ds, honeypot_ds,
                 rep_layers, w_benign, w_refusal, w_harmful, w_honeypot,
                 w_align_benign, w_align_refusal, w_align_honeypot,
                 margin_ref_gt_hp, margin_hp_gt_harm,
                 w_margin_ref_hp, w_margin_hp_harm,
                 margin_harmful_ce, w_margin_harmful_ce, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benign_ds = benign_ds
        self.refusal_ds = refusal_ds
        self.harmful_ds = harmful_ds
        self.honeypot_ds = honeypot_ds
        self.rep_layers = rep_layers
        self.w_benign = w_benign
        self.w_refusal = w_refusal
        self.w_harmful = w_harmful
        self.w_honeypot = w_honeypot
        self.w_align_benign = w_align_benign
        self.w_align_refusal = w_align_refusal
        self.w_align_honeypot = w_align_honeypot
        self.margin_ref_gt_hp = margin_ref_gt_hp
        self.margin_hp_gt_harm = margin_hp_gt_harm
        self.w_margin_ref_hp = w_margin_ref_hp
        self.w_margin_hp_harm = w_margin_hp_harm
        self.margin_harmful_ce = margin_harmful_ce
        self.w_margin_harmful_ce = w_margin_harmful_ce

        self.ema = {k: None for k in [
            "benign_ce","refusal_ce","harmful_ce","honeypot_ce",
            "align_benign","align_refusal","align_honeypot","total",
            "gap_ref_hp","gap_hp_harm","margin_harmful_ce"
        ]}

    def get_train_dataloader(self):
        args = self.args
        benign_loader = DataLoader(self.benign_ds, batch_size=args.per_device_train_batch_size, shuffle=True)
        refusal_loader = DataLoader(self.refusal_ds, batch_size=args.per_device_train_batch_size, shuffle=True)
        harmful_loader = DataLoader(self.harmful_ds, batch_size=args.per_device_train_batch_size, shuffle=True)
        honeypot_loader = DataLoader(self.honeypot_ds, batch_size=args.per_device_train_batch_size, shuffle=True)
        return QuadBatchLoader(benign_loader, refusal_loader, harmful_loader, honeypot_loader)

    def _ema_update(self, key, value, alpha=0.9):
        v = value.item() if torch.is_tensor(value) else float(value)
        if math.isnan(v): return
        if self.ema[key] is None or math.isnan(self.ema[key]): self.ema[key] = v
        else: self.ema[key] = alpha*self.ema[key] + (1-alpha)*v

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        batch_benign, batch_refusal, batch_harmful, batch_honeypot = inputs

        out_benign = model(**{k:v.to(model.device) for k,v in batch_benign.items()}, output_hidden_states=True, use_cache=False)
        loss_benign = ce_loss_from_outputs(out_benign.logits, batch_benign["labels"].to(model.device))
        align_benign = rep_alignment_loss(model, batch_benign, self.rep_layers)

        out_refusal = model(**{k:v.to(model.device) for k,v in batch_refusal.items()}, output_hidden_states=True, use_cache=False)
        loss_refusal = ce_loss_from_outputs(out_refusal.logits, batch_refusal["labels"].to(model.device))
        align_refusal = rep_alignment_loss(model, batch_refusal, self.rep_layers)

        out_harmful = model(**{k:v.to(model.device) for k,v in batch_harmful.items()}, use_cache=False)
        loss_harmful = ce_loss_from_outputs(out_harmful.logits, batch_harmful["labels"].to(model.device))

        out_honeypot = model(**{k:v.to(model.device) for k,v in batch_honeypot.items()}, use_cache=False)
        loss_honeypot = ce_loss_from_outputs(out_honeypot.logits, batch_honeypot["labels"].to(model.device))
        
        # New align_honeypot function
        align_honeypot_loss = align_honeypot(model, batch_benign, batch_honeypot, self.rep_layers)

        s_refusal = mean_logprob_per_example(out_refusal.logits, batch_refusal["labels"].to(model.device))
        s_honeypot = mean_logprob_per_example(out_honeypot.logits, batch_honeypot["labels"].to(model.device))
        s_harmful = mean_logprob_per_example(out_harmful.logits, batch_harmful["labels"].to(model.device))

        L_ref_vs_hp = F.relu(self.margin_ref_gt_hp - (s_refusal - s_honeypot)).mean()
        L_hp_vs_harm = F.relu(self.margin_hp_gt_harm - (s_honeypot - s_harmful)).mean()
        
        # Harmful CE margin term
        L_harmful_ce_margin = F.relu(self.margin_harmful_ce - loss_harmful).mean()

        total_loss = (
            self.w_benign*loss_benign +
            self.w_refusal*loss_refusal +
            self.w_harmful*loss_harmful +
            self.w_honeypot*loss_honeypot +
            self.w_align_benign*align_benign +
            self.w_align_refusal*align_refusal +
            self.w_align_honeypot*align_honeypot_loss +
            self.w_margin_ref_hp*L_ref_vs_hp +
            self.w_margin_hp_harm*L_hp_vs_harm +
            self.w_margin_harmful_ce*L_harmful_ce_margin
        )

        gap_ref_hp = (s_refusal - s_honeypot).mean()
        gap_hp_harm = (s_honeypot - s_harmful).mean()
        to_bits = 1.0 / math.log(2)

        for k,v in [
            ("benign_ce",loss_benign),("refusal_ce",loss_refusal),
            ("harmful_ce",loss_harmful),("honeypot_ce",loss_honeypot),
            ("align_benign",align_benign),("align_refusal",align_refusal),
            ("align_honeypot",align_honeypot_loss),("margin_harmful_ce",L_harmful_ce_margin),
            ("total",total_loss),("gap_ref_hp",gap_ref_hp),("gap_hp_harm",gap_hp_harm)
        ]: self._ema_update(k,v)

        self.log({
            "loss/benign_ce": loss_benign.item(),
            "loss/refusal_ce": loss_refusal.item(),
            "loss/harmful_ce": loss_harmful.item(),
            "loss/honeypot_ce": loss_honeypot.item(),
            "align/benign_l2": align_benign.item(),
            "align/refusal_l2": align_refusal.item(),
            "align/honeypot_l2": align_honeypot_loss.item(),
            "loss/margin_ref_hp": L_ref_vs_hp.item(),
            "loss/margin_hp_harm": L_hp_vs_harm.item(),
            "loss/margin_harmful_ce": L_harmful_ce_margin.item(),
            "gap/ref_hp_bits": gap_ref_hp.item()*to_bits,
            "gap/hp_harm_bits": gap_hp_harm.item()*to_bits,
            "ema/gap_ref_hp_bits": self.ema["gap_ref_hp"]*to_bits,
            "ema/gap_hp_harm_bits": self.ema["gap_hp_harm"]*to_bits,
            "ema/margin_harmful_ce": self.ema["margin_harmful_ce"],
            "loss/total": total_loss.item(),
            "ema/total": self.ema["total"]
        })
        return (total_loss, None) if return_outputs else total_loss

# -----------------------------
# Logging callback
# -----------------------------
class LoggingCallback(TrainerCallback):
    """Logs all metrics to CSV and creates plots at the end."""
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.csv_path = self.output_dir / "metrics.csv"
        self.all_logs = []
        self.fieldnames = ["step"]

    def _write_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in self.all_logs:
                writer.writerow({k: row.get(k, "") for k in self.fieldnames})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        row = {"step": state.global_step}
        row.update(logs)
        # Update header dynamically
        for k in row.keys():
            if k not in self.fieldnames:
                self.fieldnames.append(k)
        self.all_logs.append(row)
        self._write_csv()

    def on_train_end(self, args, state, control, **kwargs):
        if not self.all_logs: return
        import pandas as pd
        df = pd.DataFrame(self.all_logs)
        plt.figure(figsize=(14, 8))
        for col in df.columns:
            if col != "step" and df[col].dtype != object:
                plt.plot(df["step"], df[col], label=col)
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(self.output_dir / "metrics.png")
        print(f"📊 Saved metrics to {self.output_dir}/metrics.csv and metrics.png")

# -----------------------------
# 6. Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--ultrachat_samples", type=int, default=5000)
    parser.add_argument("--cb_path", type=str, required=True)
    parser.add_argument("--honeypot_path", type=str, required=True)
    parser.add_argument("--limit_cb", type=int, default=5000)
    parser.add_argument("--limit_hp", type=int, default=5000)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--rep_layers", type=int, nargs="*", default=[10,20,30])
    parser.add_argument("--w_benign", type=float, default=DEFAULT_W_BENIGN)
    parser.add_argument("--w_refusal", type=float, default=DEFAULT_W_REFUSAL)
    parser.add_argument("--w_harmful", type=float, default=DEFAULT_W_HARMFUL)
    parser.add_argument("--w_honeypot", type=float, default=DEFAULT_W_HONEYPOT)
    parser.add_argument("--w_align_benign", type=float, default=DEFAULT_W_ALIGN_BENIGN)
    parser.add_argument("--w_align_refusal", type=float, default=DEFAULT_W_ALIGN_REFUSAL)
    parser.add_argument("--w_align_honeypot", type=float, default=DEFAULT_W_ALIGN_HONEYPOT)
    parser.add_argument("--margin_ref_gt_hp", type=float, default=DEFAULT_MARGIN_REF_GT_HP)
    parser.add_argument("--margin_hp_gt_harm", type=float, default=DEFAULT_MARGIN_HP_GT_HARM)
    parser.add_argument("--w_margin_ref_hp", type=float, default=DEFAULT_W_MARGIN_REF_HP)
    parser.add_argument("--w_margin_hp_harm", type=float, default=DEFAULT_W_MARGIN_HP_HARM)
    parser.add_argument("--margin_harmful_ce", type=float, default=DEFAULT_MARGIN_HARMFUL_CE)
    parser.add_argument("--w_margin_harmful_ce", type=float, default=DEFAULT_W_MARGIN_HARMFUL_CE)
    parser.add_argument("--output_dir", type=str, default="./honeypot_hinge")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--grad_accum", type=int, default=2)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    safe_prompts, safe_responses = load_ultrachat(args.ultrachat_samples)
    benign_data = tokenize_chat_generic(safe_prompts, safe_responses, tokenizer, max_length=args.max_length)
    refusal_prompts, refusal_outputs = load_circuit_breakers_refusal(args.cb_path, args.limit_cb)
    refusal_data = tokenize_chat_generic(refusal_prompts, refusal_outputs, tokenizer, max_length=args.max_length)
    harmful_prompts, harmful_outputs = load_circuit_breakers_harmful(args.cb_path, args.limit_cb)
    harmful_data = tokenize_chat_generic(harmful_prompts, harmful_outputs, tokenizer, max_length=args.max_length)
    hp_prompts, hp_outputs = load_honeypots(args.honeypot_path, args.limit_hp)
    honeypot_data = tokenize_chat_generic(hp_prompts, hp_outputs, tokenizer, max_length=args.max_length)

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)
    lora_config = LoraConfig(
        r=32, lora_alpha=16, target_modules=["q_proj","v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.to(args.device)
    model.train()

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

    trainer = QuadBatchTrainer(
        model=model,
        benign_ds=ChatDataset(benign_data),
        refusal_ds=ChatDataset(refusal_data),
        harmful_ds=ChatDataset(harmful_data),
        honeypot_ds=ChatDataset(honeypot_data),
        rep_layers=args.rep_layers,
        w_benign=args.w_benign,
        w_refusal=args.w_refusal,
        w_harmful=args.w_harmful,
        w_honeypot=args.w_honeypot,
        w_align_benign=args.w_align_benign,
        w_align_refusal=args.w_align_refusal,
        w_align_honeypot=args.w_align_honeypot,
        margin_ref_gt_hp=args.margin_ref_gt_hp,
        margin_hp_gt_harm=args.margin_hp_gt_harm,
        w_margin_ref_hp=args.w_margin_ref_hp,
        w_margin_hp_harm=args.w_margin_hp_harm,
        margin_harmful_ce=args.margin_harmful_ce,
        w_margin_harmful_ce=args.w_margin_harmful_ce,
        args=training_args,
        tokenizer=tokenizer,
        callbacks=[LoggingCallback(args.output_dir)]
    )

    trainer.train()
    model.save_pretrained(os.path.join(args.output_dir, "lora_adapter"))

if __name__ == "__main__":
    main()
