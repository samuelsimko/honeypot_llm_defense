#!/usr/bin/env python3
"""
attacks/universal_embedding_attack.py

Universal soft-prompt (embedding) attack.

- initialize(): trains ONE universal embedding
- run_example(): applies frozen embedding to prompts

Prefix-style only: [OPTIM][CHAT_PROMPT][GENERATION]

No judges, no training loops outside this class.
"""

from __future__ import annotations
import time
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import torch
import torch.nn.functional as F
import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from attacks.base import Attack

try:
    from peft import PeftModel
except Exception:
    PeftModel = tuple()  # type: ignore


# ============================================================
# Config
# ============================================================

@dataclass
class UniversalEmbeddingConfig:
    lr: float = 5e-3
    num_steps: int = 2000
    num_tokens: int = 2
    optim_str_init: str = "x" * 2
    rand_init: bool = False
    seed: Optional[int] = None

    # signed weights
    w_circuit_breakers: Optional[float] = 1.0   # minimize CE
    w_honeypots: Optional[float] = None         # push CE up

    log_every: int = 50

    # generation
    max_new_tokens: int = 800

    # safety
    require_lora: bool = False

    # keep only the first N chars of the target if not None
    target_max_length: Optional[int] = None


# ============================================================
# Universal embedding attack
# ============================================================

class UniversalEmbeddingAttack(Attack):
    """
    Universal prefix embedding attack.
    """

    name = "universal_soft_prompt"

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None,
        circuit_breakers: Optional[List[Dict[str, str]]] = None,
        honeypots: Optional[List[Dict[str, str]]] = None,
        **attack_config,
    ):
        super().__init__(model, tokenizer, device=device, **attack_config)

        self.config = UniversalEmbeddingConfig(**self.attack_config)

        self.using_lora = isinstance(self.model, PeftModel)
        logging.warning(
            "[universal] using_lora=%s model_cls=%s",
            self.using_lora,
            type(self.model).__name__,
        )

        if self.config.require_lora and not self.using_lora:
            raise RuntimeError(
                "UniversalEmbeddingAttack: require_lora=True but model is not LoRA."
            )

        self.cb_data = circuit_breakers or []
        self.hp_data = honeypots or []

        self.optim_embeds: Optional[torch.Tensor] = None  # stored on CPU

    # ---------------------------------------------------------
    # helpers
    # ---------------------------------------------------------

    def _embed_device(self) -> torch.device:
        emb = self.model.get_input_embeddings()
        return emb.weight.device

    # ---------------------------------------------------------
    # training
    # ---------------------------------------------------------

    def initialize(self) -> None:
        cfg = self.config
        model = self.model
        tokenizer = self.tokenizer

        model.train()
        model.enable_input_require_grads()

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        embed_layer = model.get_input_embeddings()
        emb_device = self._embed_device()

        # ---- init embedding ----
        if cfg.rand_init:
            optim_embeds = torch.randn(
                (1, cfg.num_tokens, model.config.hidden_size),
                device=emb_device,
                requires_grad=True,
            )
        else:
            ids = tokenizer(
                cfg.optim_str_init,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"].to(emb_device)
            optim_embeds = embed_layer(ids).detach().clone()
            optim_embeds = optim_embeds[:, : cfg.num_tokens, :]
            optim_embeds.requires_grad_(True)

        opt = torch.optim.Adam([optim_embeds], lr=cfg.lr)

        # ---- sanity checks ----
        if cfg.w_circuit_breakers and not self.cb_data:
            raise ValueError("w_circuit_breakers set but no circuit breakers provided")
        if cfg.w_honeypots and not self.hp_data:
            raise ValueError("w_honeypots set but no honeypots provided")

        import random

        # ---- training loop (JOINT updates) ----
        for step in tqdm.trange(cfg.num_steps, desc="Training universal embedding"):
            losses = []

            # =====================
            # Circuit breaker term
            # =====================
            if cfg.w_circuit_breakers:
                ex = random.choice(self.cb_data)

                prompt = ex["prompt"]
                target = ex.get("response") or ex.get("output") or ex.get("target")

                # keep only the first N chars of the target if not None
                if cfg.target_max_length is not None:
                    target = target[:cfg.target_max_length]

                if target is not None:
                    messages = [{"role": "user", "content": prompt}]
                    template = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
                        template = template.replace(tokenizer.bos_token, "", 1)

                    prompt_ids = tokenizer(
                        template,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["input_ids"].to(emb_device)

                    target_ids = tokenizer(
                        target,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["input_ids"].to(emb_device)

                    prompt_embeds = embed_layer(prompt_ids)
                    target_embeds = embed_layer(target_ids)

                    train_embeds = torch.cat(
                        [optim_embeds, prompt_embeds, target_embeds.detach()],
                        dim=1,
                    )

                    attn = torch.ones(
                        train_embeds.shape[:2], device=emb_device, dtype=torch.long
                    )

                    logits = model(
                        inputs_embeds=train_embeds,
                        attention_mask=attn,
                        use_cache=False,
                    ).logits

                    shift = optim_embeds.shape[1] + prompt_ids.shape[1]
                    shift_logits = logits[
                        :, shift - 1 : shift - 1 + target_ids.shape[1], :
                    ]

                    ce_cb = F.cross_entropy(
                        shift_logits.reshape(-1, shift_logits.size(-1)),
                        target_ids.reshape(-1),
                    )

                    losses.append(float(cfg.w_circuit_breakers) * ce_cb)

            # =====================
            # Honeypot term
            # =====================
            if cfg.w_honeypots:
                ex = random.choice(self.hp_data)

                prompt = ex["prompt"]
                target = ex.get("response") or ex.get("output") or ex.get("target")

                # keep only the first N chars
                if cfg.target_max_length is not None:
                    target = target[:cfg.target_max_length]

                if target is not None:
                    messages = [{"role": "user", "content": prompt}]
                    template = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
                        template = template.replace(tokenizer.bos_token, "", 1)

                    prompt_ids = tokenizer(
                        template,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["input_ids"].to(emb_device)

                    target_ids = tokenizer(
                        target,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["input_ids"].to(emb_device)

                    prompt_embeds = embed_layer(prompt_ids)
                    target_embeds = embed_layer(target_ids)

                    train_embeds = torch.cat(
                        [optim_embeds, prompt_embeds, target_embeds.detach()],
                        dim=1,
                    )

                    attn = torch.ones(
                        train_embeds.shape[:2], device=emb_device, dtype=torch.long
                    )

                    logits = model(
                        inputs_embeds=train_embeds,
                        attention_mask=attn,
                        use_cache=False,
                    ).logits

                    shift = optim_embeds.shape[1] + prompt_ids.shape[1]
                    shift_logits = logits[
                        :, shift - 1 : shift - 1 + target_ids.shape[1], :
                    ]

                    ce_hp = F.cross_entropy(
                        shift_logits.reshape(-1, shift_logits.size(-1)),
                        target_ids.reshape(-1),
                    )

                    # discourage honeypot learning
                    losses.append(torch.relu(5.0 - ce_hp))

            if not losses:
                continue

            loss = sum(losses)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % cfg.log_every == 0:
                logging.info(
                    "[universal] step=%d loss=%.4f norm=%.3f",
                    step,
                    loss.item(),
                    optim_embeds.norm().item(),
                )

        self.optim_embeds = optim_embeds.detach().cpu()
        model.eval()

    # ---------------------------------------------------------
    # inference
    # ---------------------------------------------------------

    @torch.no_grad()
    def run_example(
        self,
        *,
        behavior_id: str,
        prompt: str,
        variant_id: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:

        assert self.optim_embeds is not None, "Call initialize() first."

        if seed is not None:
            torch.manual_seed(seed)

        model = self.model
        tokenizer = self.tokenizer
        model.eval()

        emb_device = self._embed_device()
        embed_layer = model.get_input_embeddings()

        messages = [{"role": "user", "content": prompt}]
        template = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "", 1)

        ids = tokenizer(
            template,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(emb_device)

        base_embeds = embed_layer(ids)
        optim = self.optim_embeds.to(emb_device, dtype=base_embeds.dtype)

        input_embeds = torch.cat([optim, base_embeds], dim=1)

        attn = torch.ones(
            input_embeds.shape[:2], device=emb_device, dtype=torch.long
        )

        output_ids = model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attn,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        gen_text = tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()

        return {
            "prompt": prompt,
            "generated": gen_text,
            "attack_metadata": {
                "behavior_id": behavior_id,
                "variant_id": variant_id,
                "attack_type": "universal_soft_prompt",
                "using_lora": bool(self.using_lora),
            },
        }
