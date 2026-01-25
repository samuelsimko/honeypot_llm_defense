#!/usr/bin/env python3
"""
honeypot_triplet.py

Triplet-based defense training.
FULLY ALIGNED with honeypot_cb infrastructure.

Safe   = benign + refusal
Unsafe = harmful + honeypot
Retain = benign (KL)
DPO-style: prefer honeypot over harmful
"""

# ============================================================
# IMPORT SHARED INFRASTRUCTURE FROM honeypot_cb
# ============================================================

from defenses.honeypot_cb import (
    tokenize_chat_generic,
    ChatDataset,
    load_ultrachat,
    load_circuit_breakers_harmful,
    load_circuit_breakers_refusal,
    load_honeypots,
    ce_loss_from_outputs,
    mean_logprob_per_example,
    adapters_disabled_ctx,
)

# ============================================================
# STANDARD IMPORTS (same as honeypot_cb)
# ============================================================

import os
import json
import math
import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


# ============================================================
# Defaults (paper-aligned, explicit)
# ============================================================

DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 2e-4
DEFAULT_NUM_EPOCHS = 1

DEFAULT_ALPHA = 0.5   # safe triplet
DEFAULT_NUM_MAX_STEPS = 1500
DEFAULT_BETA  = 0.4   # unsafe triplet
DEFAULT_GAMMA = 0.9   # KL

DEFAULT_W_HONEYPOT = 0.1
DEFAULT_MARGIN_HONEYPOT = 500

DEFAULT_MB = 500.0
DEFAULT_MH = 1500.0

DEFAULT_REP_LAYERS = list(range(20, 31))


def mean_logprob_per_example_from_logits(logits, labels):
    # labels: [B, T] with -100 mask
    logp = F.log_softmax(logits, dim=-1)  # [B, T, V]
    # gather token logprobs
    token_ids = labels.clamp(min=0)       # avoid -100 indexing
    tok_logp = logp.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]
    mask = (labels != -100).float()
    denom = mask.sum(dim=-1).clamp(min=1.0)
    return (tok_logp * mask).sum(dim=-1) / denom  # [B]

# ============================================================
# Representation helpers (triplet-specific)
# ============================================================

def get_hidden_states(model, batch, layers):
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
        use_cache=False,
    )
    hs = out.hidden_states
    return torch.stack([hs[l + 1] for l in layers])


def masked_mean(h, mask):
    mask = mask.unsqueeze(0).unsqueeze(-1)
    return (h * mask).sum(dim=(1,2)) / mask.sum(dim=(1,2)).clamp_min(1.0)


def d_mix(x, y):
    xn = F.normalize(x, dim=-1)
    yn = F.normalize(y, dim=-1)
    return (
        torch.norm(x - y, dim=-1)
        + 10.0 * (1 - F.cosine_similarity(xn, yn, dim=-1).relu())
    )


# ============================================================
# Trainer
# ============================================================

class TripletTrainer(Trainer):
    def __init__(
        self,
        benign_ds,
        refusal_ds,
        harmful_ds,
        honeypot_ds,
        rep_layers,
        alpha,
        beta,
        gamma,
        mb,
        mh,
        w_honeypot,
        margin_honeypot,
        w_dpo_hp_over_harm,
        dpo_beta,
        dpo_margin,
        dpo_detach_harm,
        margin_ce,
        w_margin_ce,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.benign_ds = benign_ds
        self.refusal_ds = refusal_ds
        self.harmful_ds = harmful_ds
        self.honeypot_ds = honeypot_ds

        self.rep_layers = rep_layers
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mb = mb
        self.mh = mh
        self.margin_honeypot = margin_honeypot
        self.w_honeypot = w_honeypot

        self.w_dpo_hp_over_harm = w_dpo_hp_over_harm
        self.dpo_beta = dpo_beta
        self.dpo_margin = dpo_margin
        self.dpo_detach_harm = dpo_detach_harm
        self.margin_ce = margin_ce
        self.w_margin_ce = w_margin_ce

    def get_train_dataloader(self):
        args = self.args
        return zip(
            DataLoader(self.benign_ds,  batch_size=args.per_device_train_batch_size, shuffle=True),
            DataLoader(self.refusal_ds, batch_size=args.per_device_train_batch_size, shuffle=True),
            DataLoader(self.harmful_ds, batch_size=args.per_device_train_batch_size, shuffle=True),
            DataLoader(self.honeypot_ds,batch_size=args.per_device_train_batch_size, shuffle=True),
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        benign, refusal, harmful, honeypot = inputs
        device = model.device

        benign  = {k:v.to(device) for k,v in benign.items()}
        refusal = {k:v.to(device) for k,v in refusal.items()}
        harmful = {k:v.to(device) for k,v in harmful.items()}
        honeypot= {k:v.to(device) for k,v in honeypot.items()}

        # SAFE = benign + refusal + honeypot
        safe = benign

        h_safe   = get_hidden_states(model, safe, self.rep_layers)
        h_unsafe = get_hidden_states(model, harmful, self.rep_layers)
        h_honeypot = get_hidden_states(model, honeypot, self.rep_layers)

        with adapters_disabled_ctx(model):
            h_safe_base   = get_hidden_states(model, safe, self.rep_layers)
            h_unsafe_base = get_hidden_states(model, harmful, self.rep_layers)
            h_honeypot_base = get_hidden_states(model, honeypot, self.rep_layers)

        mask_safe   = (safe["labels"]   != -100)
        mask_unsafe = (harmful["labels"]!= -100)
        mask_honeypot = (honeypot["labels"]!= -100)

        unsafe_centroid = masked_mean(h_unsafe, mask_unsafe).unsqueeze(1).unsqueeze(2)
        honeypot_centroid = masked_mean(h_honeypot, mask_honeypot).unsqueeze(1).unsqueeze(2)

        # Triplet losses
        dp = d_mix(h_safe, h_safe_base)
        dn = d_mix(h_safe, unsafe_centroid)
        L_safe = (F.relu(dp - dn + self.mb) * mask_safe.unsqueeze(0)).mean()

        dp_u = d_mix(h_unsafe, unsafe_centroid)
        dn_u = d_mix(h_unsafe, h_unsafe_base)
        L_unsafe = (F.relu(dp_u - dn_u + self.mh) * mask_unsafe.unsqueeze(0)).mean()

        # Make a triplet loss for the honeypot, similar to unsafe.
        dp_h = d_mix(h_honeypot, h_honeypot_base)
        dn_h = d_mix(h_honeypot, honeypot_centroid)
        L_honeypot = (F.relu(dp_h - dn_h + self.mh) * mask_honeypot.unsqueeze(0)).mean()

        # -----------------------------
        # Forward passes for logits (NEEDED for DPO + CE margin)
        # -----------------------------
        out_safe = model(**safe, use_cache=False)
        out_harm = model(**harmful, use_cache=False)
        out_hp   = model(**honeypot, use_cache=False)

        # token CE (scalar)
        loss_ce_safe = ce_loss_from_outputs(out_safe.logits, safe["labels"]).mean()
        loss_ce_harm = ce_loss_from_outputs(out_harm.logits, harmful["labels"]).mean()
        loss_ce_hp   = ce_loss_from_outputs(out_hp.logits, honeypot["labels"]).mean()

        # -----------------------------
        # DPO-style preference (honeypot > harmful), reference-free
        # -----------------------------
        dpo_loss = torch.zeros((), device=device)
        delta_s  = torch.zeros((), device=device)
        p_hp_pref = torch.zeros((), device=device)

        if self.w_dpo_hp_over_harm > 0.0:
            s_hp = mean_logprob_per_example_from_logits(out_hp.logits, honeypot["labels"])
            if self.dpo_detach_harm:
                with torch.no_grad():
                    s_h = mean_logprob_per_example_from_logits(out_harm.logits, harmful["labels"])
            else:
                s_h = mean_logprob_per_example_from_logits(out_harm.logits, harmful["labels"])

            delta = s_hp - s_h                          # [B]
            dpo_vec = -F.logsigmoid(self.dpo_beta * (delta - self.dpo_margin))
            dpo_loss = dpo_vec.mean()

            delta_s = delta.mean()
            p_hp_pref = torch.sigmoid(self.dpo_beta * (delta - self.dpo_margin)).mean()

        # -----------------------------
        # CE-threshold margins (same spirit as CB margin_ce)
        # Encourage both harmful and honeypot to have CE >= margin_ce
        # -----------------------------
        harmful_ce_margin = F.relu(self.margin_ce - loss_ce_harm)
        honeypot_ce_margin = F.relu(self.margin_ce - loss_ce_hp)
        ce_margin_loss = self.w_margin_ce * (harmful_ce_margin + honeypot_ce_margin)

        # -----------------------------
        # KL retain (as you had it) — but reuse out_safe to avoid extra forward
        # -----------------------------
        with adapters_disabled_ctx(model):
            base_logits = model(**safe, use_cache=False).logits

        lora_logits = out_safe.logits

        kl = F.kl_div(
            F.log_softmax(lora_logits, dim=-1),
            F.softmax(base_logits, dim=-1),
            reduction="batchmean",
        )

        # -----------------------------
        # Total loss
        # -----------------------------
        loss = (
            self.alpha * L_safe
            + self.beta * L_unsafe
            + self.gamma * kl
            + self.w_honeypot * L_honeypot
            + self.w_dpo_hp_over_harm * dpo_loss
            + ce_margin_loss
        )

        self.log({
            "loss/safe_triplet": L_safe.item(),
            "loss/unsafe_triplet": L_unsafe.item(),
            "loss/lm_loss_safe": loss_ce_safe.item(),
            "loss/lm_loss_unsafe": loss_ce_harm.item(),
            "loss/lm_loss_honeypot": loss_ce_hp.item(),
            "loss/honeypot_triplet": L_honeypot.item(),
            "loss/dpo_hp_over_harm": dpo_loss.item(),
            "loss/ce_margin_loss": ce_margin_loss.item(),
            "loss/kl": kl.item(),
            "loss/total": loss.item(),
        })

        return loss


# ============================================================
# Main (STRUCTURALLY IDENTICAL TO honeypot_cb)
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cb_path", required=True)
    parser.add_argument("--num_max_steps", type=int, default=DEFAULT_NUM_MAX_STEPS)
    parser.add_argument("--honeypot_path", required=True)
    parser.add_argument("--ultrachat_samples", type=int, default=5000)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--w_honeypot", required=False, type=float, default=DEFAULT_W_HONEYPOT)
    parser.add_argument("--margin_honeypot", required=False, type=float, default=DEFAULT_MARGIN_HONEYPOT)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--w_dpo_hp_over_harm", type=float, default=0.0)
    parser.add_argument("--dpo_beta", type=float, default=5.0)
    parser.add_argument("--dpo_margin", type=float, default=0.0)
    parser.add_argument("--dpo_detach_harm", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--margin_ce", type=float, default=10.0)
    parser.add_argument("--w_margin_ce", type=float, default=1.0)

    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    safe_prompts, safe_responses = load_ultrachat(args.ultrachat_samples)
    benign_data = tokenize_chat_generic(safe_prompts, safe_responses, tok, DEFAULT_MAX_LENGTH)

    ref_p, ref_r = load_circuit_breakers_refusal(args.cb_path, None)
    refusal_data = tokenize_chat_generic(ref_p, ref_r, tok, DEFAULT_MAX_LENGTH)

    harm_p, harm_r = load_circuit_breakers_harmful(args.cb_path, None)
    harmful_data = tokenize_chat_generic(harm_p, harm_r, tok, DEFAULT_MAX_LENGTH)

    hp_p, hp_r = load_honeypots(args.honeypot_path, None)
    honeypot_data = tokenize_chat_generic(hp_p, hp_r, tok, DEFAULT_MAX_LENGTH)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model = get_peft_model(
        model,
        LoraConfig(
            r=32,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj","v_proj"],
            task_type="CAUSAL_LM",
        )
    )

    model.train()

    trainer = TripletTrainer(
        model=model,
        benign_ds=ChatDataset(benign_data),
        refusal_ds=ChatDataset(refusal_data),
        harmful_ds=ChatDataset(harmful_data),
        honeypot_ds=ChatDataset(honeypot_data),
        rep_layers=DEFAULT_REP_LAYERS,
        alpha=DEFAULT_ALPHA,
        beta=DEFAULT_BETA,
        gamma=DEFAULT_GAMMA,
        mb=DEFAULT_MB,
        mh=DEFAULT_MH,
        margin_honeypot=DEFAULT_MARGIN_HONEYPOT if args.margin_honeypot is None else args.margin_honeypot,
        w_dpo_hp_over_harm=args.w_dpo_hp_over_harm,
        dpo_beta=args.dpo_beta,
        dpo_margin=args.dpo_margin,
        dpo_detach_harm=args.dpo_detach_harm,
        margin_ce=args.margin_ce,
        w_margin_ce=args.w_margin_ce,
        w_honeypot=args.w_honeypot,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            num_train_epochs=args.num_epochs,
            max_steps=args.num_max_steps,
            bf16=True,
            logging_steps=10,
            save_steps=200,
            report_to="wandb",
            run_name="honeypot-triplet",
        ),
        tokenizer=tok,
    )

    trainer.train()
    model.save_pretrained(out / "lora_adapter")

    (out / "READY").touch()


if __name__ == "__main__":
    main()