#!/usr/bin/env python3
"""
relative_baseline.py

Relative-baseline defense training.
STRUCTURALLY IDENTICAL to honeypot_triplet.py.

Uses adapters_disabled_ctx(model) as the reference behavior.
No custom training loop, no manual device handling.
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
    adapters_disabled_ctx,
)

# ============================================================
# STANDARD IMPORTS (same as triplet)
# ============================================================

import argparse
from pathlib import Path

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
# Defaults
# ============================================================

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 2e-4
DEFAULT_NUM_EPOCHS = 1
DEFAULT_NUM_MAX_STEPS = 1500

# relative factors
DEFAULT_K1_HARM = 5.0
DEFAULT_K2_HP = 1.0

# weights
DEFAULT_W_BENIGN = 1.0
DEFAULT_W_REFUSAL = 1.0
DEFAULT_W_HARM = 1.0
DEFAULT_W_HONEYPOT = 0.0
DEFAULT_W_KL = 0.2


# ============================================================
# Trainer
# ============================================================

class RelativeBaselineTrainer(Trainer):
    def __init__(
        self,
        benign_ds,
        refusal_ds,
        harmful_ds,
        honeypot_ds,
        *,
        k1_harm,
        k2_hp,
        w_benign,
        w_refusal,
        w_harm,
        w_honeypot,
        w_kl,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.benign_ds = benign_ds
        self.refusal_ds = refusal_ds
        self.harmful_ds = harmful_ds
        self.honeypot_ds = honeypot_ds

        self.k1 = k1_harm
        self.k2 = k2_hp

        self.w_benign = w_benign
        self.w_refusal = w_refusal
        self.w_harm = w_harm
        self.w_honeypot = w_honeypot
        self.w_kl = w_kl

    def get_train_dataloader(self):
        args = self.args
        return zip(
            DataLoader(self.benign_ds,  batch_size=args.per_device_train_batch_size, shuffle=True),
            DataLoader(self.refusal_ds, batch_size=args.per_device_train_batch_size, shuffle=True),
            DataLoader(self.harmful_ds, batch_size=args.per_device_train_batch_size, shuffle=True),
            DataLoader(self.honeypot_ds,batch_size=args.per_device_train_batch_size, shuffle=True),
        )

    def _ce_loss(self, model, batch):
        return ce_loss_from_outputs(
            model(**batch, use_cache=False).logits,
            batch["labels"].to(model.device),
        ).mean()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        benign, refusal, harmful, honeypot = inputs
        device = model.device

        benign   = {k: v.to(device) for k, v in benign.items()}
        refusal  = {k: v.to(device) for k, v in refusal.items()}
        harmful  = {k: v.to(device) for k, v in harmful.items()}
        honeypot = {k: v.to(device) for k, v in honeypot.items()}

        # ---------- defended losses ----------
        Lb_def  = self._ce_loss(model, benign)
        Lr_def  = self._ce_loss(model, refusal)
        Lh_def  = self._ce_loss(model, harmful)
        Lhp_def = self._ce_loss(model, honeypot)

        # ---------- base losses ----------
        with adapters_disabled_ctx(model):
            with torch.no_grad():
                Lb_base = self._ce_loss(model, benign)
                Lr_base = self._ce_loss(model, refusal)
                Lh_base = self._ce_loss(model, harmful)

        # ---------- targets ----------
        tgt_benign  = Lb_base
        tgt_refusal = Lr_base
        tgt_harm    = self.k1 * Lh_base
        tgt_hp      = (1.0 / self.k2) * Lh_base

        # ---------- squared matching ----------
        loss_benign  = (Lb_def  - tgt_benign ).pow(2)
        loss_refusal = (Lr_def  - tgt_refusal).pow(2)
        loss_harm    = (Lh_def  - tgt_harm   ).pow(2)
        loss_hp      = (Lhp_def - tgt_hp     ).pow(2)

        # ---------- KL retain on benign ----------
        loss_kl = torch.tensor(0.0, device=device)
        if self.w_kl > 0:
            with adapters_disabled_ctx(model):
                base_logits = model(**benign, use_cache=False).logits
            def_logits = model(**benign, use_cache=False).logits
            loss_kl = F.kl_div(
                F.log_softmax(def_logits, dim=-1),
                F.softmax(base_logits, dim=-1),
                reduction="batchmean",
            )

        # ---------- total ----------
        loss = (
            self.w_benign  * loss_benign
            + self.w_refusal * loss_refusal
            + self.w_harm    * loss_harm
            + self.w_honeypot* loss_hp
            + self.w_kl      * loss_kl
        )

        # ---------- logging ----------
        self.log({
            "loss/benign_sq": loss_benign.item(),
            "loss/refusal_sq": loss_refusal.item(),
            "loss/harm_sq": loss_harm.item(),
            "loss/honeypot_sq": loss_hp.item(),
            "loss/kl": loss_kl.item(),
            "loss/lb_def": Lb_def.item(),
            "loss/lr_def": Lr_def.item(),
            "loss/lh_def": Lh_def.item(),
            "loss/lhp_def": Lhp_def.item(),
            "ratio/harm": (Lh_def / (Lh_base + 1e-8)).item(),
            "ratio/honeypot_vs_harm": (Lhp_def / (Lh_base + 1e-8)).item(),
            "loss/total": loss.item(),
        })

        return loss


# ============================================================
# Main (IDENTICAL STRUCTURE TO triplet)
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cb_path", required=True)
    parser.add_argument("--honeypot_path", required=True)
    parser.add_argument("--ultrachat_samples", type=int, default=5000)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--num_max_steps", type=int, default=DEFAULT_NUM_MAX_STEPS)

    parser.add_argument("--k1_harm", type=float, default=DEFAULT_K1_HARM)
    parser.add_argument("--k2_hp", type=float, default=DEFAULT_K2_HP)

    parser.add_argument("--w_benign", type=float, default=DEFAULT_W_BENIGN)
    parser.add_argument("--w_refusal", type=float, default=DEFAULT_W_REFUSAL)
    parser.add_argument("--w_harm", type=float, default=DEFAULT_W_HARM)
    parser.add_argument("--w_honeypot", type=float, default=DEFAULT_W_HONEYPOT)
    parser.add_argument("--w_kl", type=float, default=DEFAULT_W_KL)

    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---------- data ----------
    safe_p, safe_r = load_ultrachat(args.ultrachat_samples)
    benign_ds = ChatDataset(tokenize_chat_generic(safe_p, safe_r, tok, DEFAULT_MAX_LENGTH))

    ref_p, ref_r = load_circuit_breakers_refusal(args.cb_path, None)
    refusal_ds = ChatDataset(tokenize_chat_generic(ref_p, ref_r, tok, DEFAULT_MAX_LENGTH))

    harm_p, harm_r = load_circuit_breakers_harmful(args.cb_path, None)
    harmful_ds = ChatDataset(tokenize_chat_generic(harm_p, harm_r, tok, DEFAULT_MAX_LENGTH))

    hp_p, hp_r = load_honeypots(args.honeypot_path, None)
    honeypot_ds = ChatDataset(tokenize_chat_generic(hp_p, hp_r, tok, DEFAULT_MAX_LENGTH))

    # ---------- model ----------
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
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.train()

    trainer = RelativeBaselineTrainer(
        model=model,
        benign_ds=benign_ds,
        refusal_ds=refusal_ds,
        harmful_ds=harmful_ds,
        honeypot_ds=honeypot_ds,
        k1_harm=args.k1_harm,
        k2_hp=args.k2_hp,
        w_benign=args.w_benign,
        w_refusal=args.w_refusal,
        w_harm=args.w_harm,
        w_honeypot=args.w_honeypot,
        w_kl=args.w_kl,
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
            run_name="relative-baseline",
        ),
        tokenizer=tok,
    )

    trainer.train()
    model.save_pretrained(out / "lora_adapter")
    (out / "READY").touch()


if __name__ == "__main__":
    main()