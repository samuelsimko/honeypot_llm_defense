#!/usr/bin/env python3
"""
dpo_base.py

Very simple DPO-based baseline defense.

Goals:
- Benign behavior unchanged (match base LM)
- Harmful outputs have CE >= threshold
- Honeypot outputs have CE >= threshold
- Preference: P(honeypot) > P(harmful) via DPO

INTENTIONALLY WEAK baseline.
"""

# ============================================================
# IMPORT SHARED INFRASTRUCTURE
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
    mean_logprob_per_example_from_logits,
)

# ============================================================
# STANDARD IMPORTS
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

# CE thresholds
DEFAULT_HARM_CE_MIN = 4.0
DEFAULT_HP_CE_MIN   = 4.0

# weights
DEFAULT_W_BENIGN = 1.0
DEFAULT_W_HARM   = 1.0
DEFAULT_W_HP     = 1.0
DEFAULT_W_KL     = 0.2

# DPO params
DEFAULT_DPO_BETA   = 2.0
DEFAULT_DPO_MARGIN = 0.5
DEFAULT_DPO_DETACH_HARM = True

DEFAULT_W_DPO_HP_OVER_HARM = 0.0


# ============================================================
# Trainer
# ============================================================

class DPOBaselineTrainer(Trainer):
    def __init__(
        self,
        benign_ds,
        refusal_ds,
        harmful_ds,
        honeypot_ds,
        *,
        harm_ce_min,
        hp_ce_min,
        w_benign,
        w_harm,
        w_hp,
        w_dpo_hp_over_harm,
        w_kl,
        dpo_beta,
        dpo_margin,
        dpo_detach_harm,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.benign_ds = benign_ds
        self.refusal_ds = refusal_ds
        self.harmful_ds = harmful_ds
        self.honeypot_ds = honeypot_ds

        self.harm_ce_min = harm_ce_min
        self.hp_ce_min   = hp_ce_min

        self.w_benign = w_benign
        self.w_harm   = w_harm
        self.w_hp     = w_hp
        self.w_dpo_hp_over_harm = w_dpo_hp_over_harm
        self.w_kl     = w_kl

        self.dpo_beta = dpo_beta
        self.dpo_margin = dpo_margin
        self.dpo_detach_harm = dpo_detach_harm

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
        harmful  = {k: v.to(device) for k, v in harmful.items()}
        honeypot = {k: v.to(device) for k, v in honeypot.items()}

        # --------------------------------------------------
        # CE losses (defended)
        # --------------------------------------------------
        Lb_def  = self._ce_loss(model, benign)
        Lh_def  = self._ce_loss(model, harmful)
        Lhp_def = self._ce_loss(model, honeypot)

        # --------------------------------------------------
        # Base benign (anchor)
        # --------------------------------------------------
        with adapters_disabled_ctx(model):
            with torch.no_grad():
                Lb_base = self._ce_loss(model, benign)

        # --------------------------------------------------
        # Benign anchoring
        # --------------------------------------------------
        loss_benign = (Lb_def - Lb_base).pow(2)

        # --------------------------------------------------
        # Harm / Honeypot CE floor
        # Penalize if BELOW threshold
        # --------------------------------------------------
        loss_harm = F.relu(self.harm_ce_min - Lh_def).pow(2)
        loss_hp   = F.relu(self.hp_ce_min   - Lhp_def).pow(2)

        # --------------------------------------------------
        # DPO: prefer honeypot over harmful
        # --------------------------------------------------
        dpo_loss = torch.zeros((), device=device)
        delta_s  = torch.zeros((), device=device)

        if self.w_dpo_hp_over_harm > 0:
            out_hp = model(**honeypot, use_cache=False)
            out_h  = model(**harmful, use_cache=False)

            s_hp = mean_logprob_per_example_from_logits(
                out_hp.logits, honeypot["labels"].to(device)
            )

            if self.dpo_detach_harm:
                with torch.no_grad():
                    s_h = mean_logprob_per_example_from_logits(
                        out_h.logits, harmful["labels"].to(device)
                    )
            else:
                s_h = mean_logprob_per_example_from_logits(
                    out_h.logits, harmful["labels"].to(device)
                )

            delta = s_hp - s_h
            dpo_loss = -F.logsigmoid(self.dpo_beta * (delta - self.dpo_margin)).mean()
            delta_s = delta.mean()

        # --------------------------------------------------
        # KL retain on benign
        # --------------------------------------------------
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

        # --------------------------------------------------
        # Total
        # --------------------------------------------------
        loss = (
            self.w_benign * loss_benign
            + self.w_harm * loss_harm
            + self.w_hp   * loss_hp
            + self.w_dpo_hp_over_harm  * dpo_loss
            + self.w_kl   * loss_kl
        )

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------
        self.log({
            "loss/benign_anchor": loss_benign.item(),
            "loss/harm_floor": loss_harm.item(),
            "loss/hp_floor": loss_hp.item(),
            "loss/dpo": dpo_loss.item(),
            "stats/delta_s": delta_s.item(),
            "loss/kl": loss_kl.item(),
            "loss/lb_def": Lb_def.item(),
            "loss/lh_def": Lh_def.item(),
            "loss/lhp_def": Lhp_def.item(),
            "loss/total": loss.item(),
        })

        return loss


# ============================================================
# Main
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

    parser.add_argument("--harm_ce_min", type=float, default=DEFAULT_HARM_CE_MIN)
    parser.add_argument("--hp_ce_min", type=float, default=DEFAULT_HP_CE_MIN)

    parser.add_argument("--w_benign", type=float, default=DEFAULT_W_BENIGN)
    parser.add_argument("--w_harm", type=float, default=DEFAULT_W_HARM)
    parser.add_argument("--w_hp", type=float, default=DEFAULT_W_HP)
    parser.add_argument("--w_dpo_hp_over_harm", type=float, default=DEFAULT_W_DPO_HP_OVER_HARM)
    parser.add_argument("--w_kl", type=float, default=DEFAULT_W_KL)

    parser.add_argument("--dpo_beta", type=float, default=DEFAULT_DPO_BETA)
    parser.add_argument("--dpo_margin", type=float, default=DEFAULT_DPO_MARGIN)
    parser.add_argument("--dpo_detach_harm", type=bool, default=DEFAULT_DPO_DETACH_HARM)


    args = parser.parse_args()

    run_name = f"dpo-base-wb{args.w_benign}-wh{args.w_harm}-whp{args.w_hp}-wdpo_hp_over_harm{args.w_dpo_hp_over_harm}-wkl{args.w_kl}-dpo_beta{args.dpo_beta}-dpo_margin{args.dpo_margin}-dpo_detach_harm{args.dpo_detach_harm}"

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---------- data ----------
    safe_p, safe_r = load_ultrachat(args.ultrachat_samples)
    benign_ds = ChatDataset(tokenize_chat_generic(safe_p, safe_r, tok, DEFAULT_MAX_LENGTH))

    harm_p, harm_r = load_circuit_breakers_harmful(args.cb_path, None)
    harmful_ds = ChatDataset(tokenize_chat_generic(harm_p, harm_r, tok, DEFAULT_MAX_LENGTH))

    hp_p, hp_r = load_honeypots(args.honeypot_path, None)
    honeypot_ds = ChatDataset(tokenize_chat_generic(hp_p, hp_r, tok, DEFAULT_MAX_LENGTH))

    # dummy refusal dataset (kept for structural parity)
    refusal_ds = harmful_ds

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

    trainer = DPOBaselineTrainer(
        model=model,
        benign_ds=benign_ds,
        refusal_ds=refusal_ds,
        harmful_ds=harmful_ds,
        honeypot_ds=honeypot_ds,
        harm_ce_min=args.harm_ce_min,
        hp_ce_min=args.hp_ce_min,
        w_benign=args.w_benign,
        w_harm=args.w_harm,
        w_hp=args.w_hp,
        w_dpo_hp_over_harm=args.w_dpo_hp_over_harm,
        w_kl=args.w_kl,
        dpo_beta=args.dpo_beta,
        dpo_margin=args.dpo_margin,
        dpo_detach_harm=args.dpo_detach_harm,
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
            run_name=run_name,
        ),
        tokenizer=tok,
    )

    trainer.train()
    model.save_pretrained(out / "lora_adapter")
    (out / "READY").touch()


if __name__ == "__main__":
    main()
