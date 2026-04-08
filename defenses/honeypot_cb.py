import os
import json
import math
import csv
import argparse
from pathlib import Path
from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizer,
    TrainerCallback,
)

# -----------------------------
# 1. Defaults
# -----------------------------
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_EPOCHS = 1
DEFAULT_NUM_MAX_STEPS = 1500
DEFAULT_LR = 2e-4

DEFAULT_W_BENIGN = 0
DEFAULT_W_REFUSAL = 0.0
DEFAULT_W_HONEYPOT = 0.0
DEFAULT_W_HARMFUL = 0.0
DEFAULT_W_BREAKING = 3.0

DEFAULT_W_ALIGN_BENIGN = 1
DEFAULT_W_ALIGN_REFUSAL = 0
DEFAULT_W_ALIGN_HONEYPOT = 0.0

DEFAULT_MARGIN_REF_GT_HP = 0.25
DEFAULT_MARGIN_HP_GT_HARM = 0.25
DEFAULT_W_MARGIN_REF_HP = 0.5
DEFAULT_W_MARGIN_HP_HARM = 0.5
DEFAULT_MARGIN_HARMFUL_CE = 0.0
DEFAULT_W_MARGIN_HARMFUL_CE = 0.0

DEFAULT_REFUSAL_RESPONSE = "I'm sorry, I cannot do that."
DEFAULT_FSDP_CONFIG = "fsdp_config.yaml"


def module_device(model) -> torch.device:
    return next(model.parameters()).device

from transformers.tokenization_utils_base import BatchEncoding

def get_input_ids(x):
    if isinstance(x, BatchEncoding):
        return x["input_ids"]
    if isinstance(x, dict):
        return x["input_ids"]
    return x

# -----------------------------
# 2. Tokenization helpers
# -----------------------------
def tokenize_chat_generic(
    prompts: List[str],
    responses: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = DEFAULT_MAX_LENGTH,
):
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must match length")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out = []
    for prompt, response in zip(prompts, responses):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        full_tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        prompt_tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        full_ids = get_input_ids(full_tokens)
        prompt_ids = get_input_ids(prompt_tokens)

        labels = full_ids.clone()
        prompt_len = prompt_ids.shape[-1]
        labels[0, :prompt_len] = -100

        L = full_ids.shape[-1]
        pad_len = max_length - L

        input_ids = F.pad(full_ids, (0, pad_len), value=tokenizer.pad_token_id).squeeze(0)
        labels = F.pad(labels, (0, pad_len), value=-100).squeeze(0)
        attn = torch.ones_like(full_ids)
        attn = F.pad(attn, (0, pad_len), value=0).squeeze(0)

        out.append(
            {
                "input_ids": input_ids,
                "attention_mask": attn,
                "labels": labels,
            }
        )
    return out


class ChatDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# -----------------------------
# 3. Data loading
# -----------------------------
def load_ultrachat(num_samples: int, seed: int = 42):
    ds = (
        load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        .shuffle(seed=seed)
        .select(range(num_samples))
    )
    prompts, responses = [], []
    for item in ds:
        messages = item["messages"]
        if not messages:
            continue
        user = next((x["content"] for x in messages if x["role"] == "user"), "")
        assistant = next((x["content"] for x in messages if x["role"] == "assistant"), "")
        prompts.append(user)
        responses.append(assistant)
    return prompts, responses


def _load_json_or_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(8192)
        f.seek(0)
        first_non_ws = next((ch for ch in head if not ch.isspace()), None)

        def parse_jsonl(file_obj):
            out = []
            for line in file_obj:
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("//"):
                    continue
                try:
                    out.append(json.loads(s))
                except json.JSONDecodeError:
                    continue
            return out

        if first_non_ws == "[":
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
            except Exception:
                f.seek(0)
                return parse_jsonl(f)
        return parse_jsonl(f)


def load_circuit_breakers_harmful(path: str, limit: Optional[int]):
    recs = _load_json_or_jsonl(path)
    if limit:
        recs = recs[:limit]
    prompts = [r.get("prompt", "") for r in recs]
    harmful = [r.get("output", "") for r in recs]
    return prompts, harmful


def load_circuit_breakers_refusal(path: str, limit: Optional[int]):
    recs = _load_json_or_jsonl(path)
    if limit:
        recs = recs[:limit]
    prompts = [r.get("prompt", "") for r in recs]
    refusals = [(r.get("llama3_output") or DEFAULT_REFUSAL_RESPONSE) for r in recs]
    return prompts, refusals


def load_honeypots(path: str, limit: Optional[int]):
    recs = _load_json_or_jsonl(path)
    if limit:
        recs = recs[:limit]
    prompts = [r.get("prompt", "") for r in recs]
    hp = [r.get("response", "") for r in recs]
    return prompts, hp


# -----------------------------
# 4. Utils
# -----------------------------
def ce_loss_from_outputs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def mean_logprob_per_example_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    gather_ids = shift_labels.clamp(min=0).unsqueeze(-1)
    token_logp = log_probs.gather(-1, gather_ids).squeeze(-1)

    mask = (shift_labels != -100).float()
    token_logp = token_logp * mask

    denom = mask.sum(dim=-1).clamp(min=1.0)
    return token_logp.sum(dim=-1) / denom


@contextmanager
def adapters_disabled_ctx(model):
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
    else:
        yield


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def rep_alignment_loss(model, batch, layer_ids):
    device = module_device(model)
    batch = move_batch(batch, device)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    out_adapt = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hs_adapt = out_adapt.hidden_states

    with adapters_disabled_ctx(model):
        with torch.no_grad():
            out_base = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hs_base = out_base.hidden_states

    num_layers = len(hs_adapt) - 1
    chosen = [max(0, min(num_layers - 1, int(lid))) + 1 for lid in layer_ids]
    mask = (labels != -100).unsqueeze(-1).float()

    losses = []
    for idx in chosen:
        a = hs_adapt[idx]
        b = hs_base[idx]
        diff = (a - b) * mask
        denom = mask.sum() * a.size(-1) + 1e-8
        losses.append(diff.pow(2).sum() / denom)

    return sum(losses) / max(1, len(losses))


def breaking_loss(model, batch_harmful, layer_ids):
    device = module_device(model)
    batch_harmful = move_batch(batch_harmful, device)

    input_ids = batch_harmful["input_ids"]
    attention_mask = batch_harmful["attention_mask"]
    labels = batch_harmful["labels"]

    out_adapt = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hs_adapt = out_adapt.hidden_states

    with adapters_disabled_ctx(model):
        with torch.no_grad():
            out_base = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hs_base = out_base.hidden_states

    num_layers = len(hs_adapt) - 1
    chosen = [max(0, min(num_layers - 1, int(lid))) + 1 for lid in layer_ids]

    mask = (labels != -100).float() * attention_mask.float()

    losses = []
    for idx in chosen:
        a = hs_adapt[idx]
        b = hs_base[idx]
        cos = F.cosine_similarity(a, b, dim=-1)
        relu_cos = F.relu(cos) * mask
        denom = mask.sum() + 1e-8
        losses.append(relu_cos.sum() / denom)

    return sum(losses) / max(1, len(losses))

def l2_alignment_from_states(hs_adapt, hs_base, labels, layer_ids):
    num_layers = len(hs_adapt) - 1
    chosen = [max(0, min(num_layers - 1, int(lid))) + 1 for lid in layer_ids]
    mask = (labels != -100).unsqueeze(-1).float()

    losses = []
    for idx in chosen:
        a = hs_adapt[idx]
        b = hs_base[idx]
        diff = (a - b) * mask
        denom = mask.sum() * a.size(-1) + 1e-8
        losses.append(diff.pow(2).sum() / denom)

    return sum(losses) / max(1, len(losses))


def breaking_from_states(hs_adapt, hs_base, labels, attention_mask, layer_ids):
    num_layers = len(hs_adapt) - 1
    chosen = [max(0, min(num_layers - 1, int(lid))) + 1 for lid in layer_ids]
    mask = (labels != -100).float() * attention_mask.float()

    losses = []
    for idx in chosen:
        a = hs_adapt[idx]
        b = hs_base[idx]
        cos = F.cosine_similarity(a, b, dim=-1)
        relu_cos = F.relu(cos) * mask
        denom = mask.sum() + 1e-8
        losses.append(relu_cos.sum() / denom)

    return sum(losses) / max(1, len(losses))


def run_base_model_hidden_states(model, batch, need_hidden_states=True):
    with adapters_disabled_ctx(model):
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=need_hidden_states,
                use_cache=False,
            )
    return out
# -----------------------------
# 5. Quad loader & trainer
# -----------------------------
class QuadBatchLoader:
    def __init__(self, benign_loader, refusal_loader, harmful_loader, honeypot_loader):
        self.benign_loader = benign_loader
        self.refusal_loader = refusal_loader
        self.harmful_loader = harmful_loader
        self.honeypot_loader = honeypot_loader
        self._len = min(
            len(benign_loader),
            len(refusal_loader),
            len(harmful_loader),
            len(honeypot_loader),
        )

    def __iter__(self):
        return zip(
            self.benign_loader,
            self.refusal_loader,
            self.harmful_loader,
            self.honeypot_loader,
        )

    def __len__(self):
        return self._len


class QuadBatchTrainer(Trainer):
    def __init__(
        self,
        benign_ds,
        refusal_ds,
        harmful_ds,
        honeypot_ds,
        rep_layers,
        w_benign,
        w_refusal,
        w_harmful,
        w_honeypot,
        w_align_benign,
        w_align_refusal,
        w_align_honeypot,
        margin_ref_gt_hp,
        margin_hp_gt_harm,
        w_margin_ref_hp,
        w_margin_hp_harm,
        w_breaking,
        margin_harmful_ce,
        w_margin_harmful_ce,
        w_dpo_hp_over_harm: float = 0.0,
        dpo_beta: float = 5.0,
        dpo_margin: float = 0.0,
        dpo_detach_harm: bool = True,
        margin_ce: float = 10.0,
        w_margin_ce: float = 1.0,
        breaking_start_mult=10.0,
        breaking_warmup_steps=200,
        *args,
        **kwargs,
    ):
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
        self.w_breaking = w_breaking

        self.w_dpo_hp_over_harm = w_dpo_hp_over_harm
        self.dpo_beta = dpo_beta
        self.dpo_margin = dpo_margin
        self.dpo_detach_harm = dpo_detach_harm

        self.margin_ce = margin_ce
        self.w_margin_ce = w_margin_ce

        self.breaking_start_mult = breaking_start_mult
        self.breaking_warmup_steps = breaking_warmup_steps

        self.ema = {
            k: None
            for k in [
                "benign_ce",
                "refusal_ce",
                "harmful_ce",
                "honeypot_ce",
                "align_benign",
                "align_refusal",
                "total",
                "breaking_harmful",
                "breaking_honeypot",
                "breaking",
                "dpo_hp_over_harm",
                "delta_s_hp_harm",
                "p_hp_pref",
            ]
        }

    def get_train_dataloader(self):
        args = self.args
        benign_loader = DataLoader(self.benign_ds, batch_size=args.per_device_train_batch_size, shuffle=True)
        refusal_loader = DataLoader(self.refusal_ds, batch_size=args.per_device_train_batch_size, shuffle=True)
        harmful_loader = DataLoader(self.harmful_ds, batch_size=args.per_device_train_batch_size, shuffle=True)
        honeypot_loader = DataLoader(self.honeypot_ds, batch_size=args.per_device_train_batch_size, shuffle=True)
        return QuadBatchLoader(benign_loader, refusal_loader, harmful_loader, honeypot_loader)

    def _ema_update(self, key, value, alpha=0.9):
        v = value.item() if torch.is_tensor(value) else float(value)
        if math.isnan(v):
            return
        if self.ema[key] is None or math.isnan(self.ema[key]):
            self.ema[key] = v
        else:
            self.ema[key] = alpha * self.ema[key] + (1 - alpha) * v


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = module_device(model)
        batch_benign, batch_refusal, batch_harmful, batch_honeypot = inputs

        batch_benign = move_batch(batch_benign, device)
        batch_refusal = move_batch(batch_refusal, device)
        batch_harmful = move_batch(batch_harmful, device)
        batch_honeypot = move_batch(batch_honeypot, device)

        total_loss = torch.zeros((), device=device)
        logs = {}
        ema_updates = []

        # Decide what actually needs to be computed
        need_benign_hidden = (self.w_benign != 0.0) or (self.w_align_benign != 0.0)
        need_refusal = (self.w_refusal != 0.0)
        need_harmful_hidden = (self.w_harmful != 0.0) or (self.w_breaking != 0.0) or (self.w_dpo_hp_over_harm != 0.0)
        need_honeypot_hidden = (self.w_honeypot != 0.0) or (self.w_breaking != 0.0) or (self.w_dpo_hp_over_harm != 0.0)

        out_benign = None
        out_refusal = None
        out_harmful = None
        out_honeypot = None

        if need_benign_hidden:
            out_benign = model(
                **batch_benign,
                output_hidden_states=(self.w_align_benign != 0.0),
                use_cache=False,
            )

        if need_refusal:
            out_refusal = model(
                **batch_refusal,
                output_hidden_states=False,
                use_cache=False,
            )

        if need_harmful_hidden:
            out_harmful = model(
                **batch_harmful,
                output_hidden_states=(self.w_breaking != 0.0),
                use_cache=False,
            )

        if need_honeypot_hidden:
            out_honeypot = model(
                **batch_honeypot,
                output_hidden_states=(self.w_breaking != 0.0),
                use_cache=False,
            )

        # CE losses
        if self.w_benign != 0.0:
            loss_benign = ce_loss_from_outputs(out_benign.logits, batch_benign["labels"])
            total_loss = total_loss + self.w_benign * loss_benign
            logs["loss/benign_ce"] = loss_benign.item()
            ema_updates.append(("benign_ce", loss_benign))
        else:
            loss_benign = None

        if self.w_refusal != 0.0:
            loss_refusal = ce_loss_from_outputs(out_refusal.logits, batch_refusal["labels"])
            total_loss = total_loss + self.w_refusal * loss_refusal
            logs["loss/refusal_ce"] = loss_refusal.item()
            ema_updates.append(("refusal_ce", loss_refusal))
        else:
            loss_refusal = None

        if self.w_harmful != 0.0 or self.w_dpo_hp_over_harm != 0.0:
            loss_harmful = ce_loss_from_outputs(out_harmful.logits, batch_harmful["labels"])
            if self.w_harmful != 0.0:
                total_loss = total_loss + self.w_harmful * loss_harmful
                logs["loss/harmful_ce"] = loss_harmful.item()
                ema_updates.append(("harmful_ce", loss_harmful))
        else:
            loss_harmful = None

        if self.w_honeypot != 0.0 or self.w_dpo_hp_over_harm != 0.0:
            loss_honeypot = ce_loss_from_outputs(out_honeypot.logits, batch_honeypot["labels"])
            if self.w_honeypot != 0.0:
                total_loss = total_loss + self.w_honeypot * loss_honeypot
                logs["loss/honeypot_ce"] = loss_honeypot.item()
                ema_updates.append(("honeypot_ce", loss_honeypot))
        else:
            loss_honeypot = None

        # Benign alignment only
        if self.w_align_benign != 0.0:
            out_benign_base = run_base_model_hidden_states(model, batch_benign, need_hidden_states=True)
            align_benign = l2_alignment_from_states(
                out_benign.hidden_states,
                out_benign_base.hidden_states,
                batch_benign["labels"],
                self.rep_layers,
            )
            total_loss = total_loss + self.w_align_benign * align_benign
            logs["align/benign_l2"] = align_benign.item()
            ema_updates.append(("align_benign", align_benign))

        # Breaking only for harmful + honeypot
        if self.w_breaking != 0.0:
            out_harmful_base = run_base_model_hidden_states(model, batch_harmful, need_hidden_states=True)
            out_honeypot_base = run_base_model_hidden_states(model, batch_honeypot, need_hidden_states=True)

            breaking_harmful_val = breaking_from_states(
                out_harmful.hidden_states,
                out_harmful_base.hidden_states,
                batch_harmful["labels"],
                batch_harmful["attention_mask"],
                self.rep_layers,
            )
            breaking_honeypot_val = breaking_from_states(
                out_honeypot.hidden_states,
                out_honeypot_base.hidden_states,
                batch_honeypot["labels"],
                batch_honeypot["attention_mask"],
                self.rep_layers,
            )
            breaking_overall = 0.5 * (breaking_harmful_val + breaking_honeypot_val)

            current_step = self.state.global_step
            if self.breaking_warmup_steps > 0:
                frac = min(1.0, current_step / self.breaking_warmup_steps)
                breaking_mult = self.breaking_start_mult - (self.breaking_start_mult - 1.0) * frac
            else:
                breaking_mult = 1.0

            effective_w_breaking = self.w_breaking * breaking_mult
            total_loss = total_loss + effective_w_breaking * breaking_overall

            logs["loss/breaking"] = breaking_overall.item()
            logs["loss/breaking_harmful"] = breaking_harmful_val.item()
            logs["loss/breaking_honeypot"] = breaking_honeypot_val.item()
            logs["sched/breaking_mult"] = breaking_mult
            logs["sched/effective_w_breaking"] = effective_w_breaking
            ema_updates.append(("breaking", breaking_overall))
            ema_updates.append(("breaking_harmful", breaking_harmful_val))
            ema_updates.append(("breaking_honeypot", breaking_honeypot_val))

        # DPO loss
        if self.w_dpo_hp_over_harm != 0.0:
            s_hp = mean_logprob_per_example_from_logits(out_honeypot.logits, batch_honeypot["labels"])

            if self.dpo_detach_harm:
                with torch.no_grad():
                    s_harm = mean_logprob_per_example_from_logits(out_harmful.logits, batch_harmful["labels"])
            else:
                s_harm = mean_logprob_per_example_from_logits(out_harmful.logits, batch_harmful["labels"])

            delta = s_hp - s_harm
            dpo_loss = -F.logsigmoid(self.dpo_beta * (delta - self.dpo_margin)).mean()

            total_loss = total_loss + self.w_dpo_hp_over_harm * dpo_loss
            logs["loss/dpo_hp_over_harm"] = dpo_loss.item()
            logs["stats/delta_s_hp_minus_harm"] = delta.mean().item()
            logs["stats/p_hp_pref"] = torch.sigmoid(self.dpo_beta * (delta - self.dpo_margin)).mean().item()

            ema_updates.append(("dpo_hp_over_harm", dpo_loss))
            ema_updates.append(("delta_s_hp_harm", delta.mean()))
            ema_updates.append(("p_hp_pref", torch.sigmoid(self.dpo_beta * (delta - self.dpo_margin)).mean()))

        # Optional floor terms if you still want them
        if loss_honeypot is not None:
            honeypot_floor = F.relu(self.margin_ce - loss_honeypot)
            total_loss = total_loss + honeypot_floor

        if loss_harmful is not None:
            harmful_floor = F.relu(self.margin_ce - loss_harmful)
            total_loss = total_loss + harmful_floor

        logs["loss/total"] = total_loss.item()
        ema_updates.append(("total", total_loss))

        for k, v in ema_updates:
            self._ema_update(k, v)

        logs["ema/total"] = self.ema["total"]
        self.log(logs)

        return (total_loss, None) if return_outputs else total_loss


# -----------------------------
# Logging callback
# -----------------------------
class LoggingCallback(TrainerCallback):
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
        if not logs:
            return
        row = {"step": state.global_step}
        row.update(logs)
        for k in row.keys():
            if k not in self.fieldnames:
                self.fieldnames.append(k)
        self.all_logs.append(row)
        self._write_csv()

    def on_train_end(self, args, state, control, **kwargs):
        if not self.all_logs:
            return
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
        print(f"Saved metrics to {self.output_dir}/metrics.csv and metrics.png")


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
    parser.add_argument("--num_epochs", "--epochs", dest="num_epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--num_max_steps", type=int, default=DEFAULT_NUM_MAX_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--rep_layers", type=int, nargs="*", default=[10, 20])

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
    parser.add_argument("--w_breaking", type=float, default=DEFAULT_W_BREAKING)

    parser.add_argument("--output_dir", type=str, default="./honeypot_cb")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--grad_accum", type=int, default=2)

    parser.add_argument("--w_dpo_hp_over_harm", type=float, default=0.5)
    parser.add_argument("--dpo_beta", type=float, default=5.0)
    parser.add_argument("--dpo_margin", type=float, default=0.0)
    parser.add_argument("--margin_ce", type=float, default=10.0)
    parser.add_argument("--fsdp_config_path", type=str, default="fsdp_config.json")
    parser.add_argument(
        "--dpo_detach_harm",
        type=lambda x: x.lower() == "true",
        default=False,
    )
    parser.add_argument("--breaking_start_mult", type=float, default=10.0)
    parser.add_argument("--breaking_warmup_steps", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    safe_prompts, safe_responses = load_ultrachat(args.ultrachat_samples)
    benign_data = tokenize_chat_generic(safe_prompts, safe_responses, tokenizer, max_length=args.max_length)

    refusal_prompts, refusal_outputs = load_circuit_breakers_refusal(args.cb_path, args.limit_cb)
    refusal_data = tokenize_chat_generic(refusal_prompts, refusal_outputs, tokenizer, max_length=args.max_length)

    harmful_prompts, harmful_outputs = load_circuit_breakers_harmful(args.cb_path, args.limit_cb)
    harmful_data = tokenize_chat_generic(harmful_prompts, harmful_outputs, tokenizer, max_length=args.max_length)

    hp_prompts, hp_outputs = load_honeypots(args.honeypot_path, args.limit_hp)
    honeypot_data = tokenize_chat_generic(hp_prompts, hp_outputs, tokenizer, max_length=args.max_length)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()

    run_name = (
        f"cb_llama"
        f"_wb{args.w_breaking}"
        f"_dpo{args.w_dpo_hp_over_harm}"
        f"_beta{args.dpo_beta}"
        f"_margin_ce{args.margin_ce}"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.num_max_steps,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_grad_norm=1.0,
        save_total_limit=args.save_total_limit,
        report_to="wandb",
        run_name=run_name,
        # ddp_find_unused_parameters=False,
        # fsdp="full_shard auto_wrap",
        # fsdp_config=args.fsdp_config_path,
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
        w_breaking=args.w_breaking,
        margin_ce=args.margin_ce,
        args=training_args,
        processing_class=tokenizer,
        w_dpo_hp_over_harm=args.w_dpo_hp_over_harm,
        dpo_beta=args.dpo_beta,
        dpo_margin=args.dpo_margin,
        dpo_detach_harm=args.dpo_detach_harm,
        callbacks=[LoggingCallback(args.output_dir)],
    )

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "lora_adapter"))

    manifest = {
        "schema_version": "1.0",
        "defense_name": "honeypot_cb",
        "run_id": output_dir.name,
        "base_model": args.model,
        "adapter_type": "lora",
        "adapter_path": "lora_adapter",
        "training_completed": True,
        "weights": {
            "w_honeypot": args.w_honeypot,
            "w_align_honeypot": args.w_align_honeypot,
            "w_breaking": args.w_breaking,
        },
        "data": {
            "ultrachat_samples": args.ultrachat_samples,
            "honeypot_path": args.honeypot_path,
            "cb_path": args.cb_path,
        },
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(output_dir / "hparams.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    (output_dir / "READY").touch()


if __name__ == "__main__":
    import torch.utils.checkpoint as checkpoint
    checkpoint.set_checkpoint_debug_enabled(True)
    main()
