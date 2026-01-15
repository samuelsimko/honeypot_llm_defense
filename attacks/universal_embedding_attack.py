"""
attacks/universal_embedding_attack.py

Universal soft-prompt (embedding) attack.

- initialize(): trains ONE embedding using CB and/or honeypots
- run_example(): applies the frozen embedding to a single prompt

No judging logic here.
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
    w_circuit_breakers: Optional[float] = 1.0    # +1 = encourage harm
    w_honeypots: Optional[float] = None           # -1 = avoid honeypots

    log_every: int = 50


# ============================================================
# Universal embedding attack
# ============================================================

class UniversalEmbeddingAttack(Attack):
    """
    Universal embedding attack.

    Contract:
      - initialize(): trains the embedding once
      - run_example(): applies embedding to a single prompt
    """

    name = "universal_soft_prompt"

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None,
        circuit_breakers: Optional[List[Dict[str, str]]] = None,
        honeypots: Optional[List[Dict[str, str]]] = None,
        max_generation_length: int = 512,
        **attack_config,
    ):
        super().__init__(model, tokenizer, device=device, **attack_config)

        # Typed config for internal use
        self.config = UniversalEmbeddingConfig(**self.attack_config)

        self.cb_data = circuit_breakers or []
        self.hp_data = honeypots or []
        self.max_generation_length = max_generation_length

        self.optim_embeds: Optional[torch.Tensor] = None

    # --------------------------------------------------------
    # Initialization = training
    # --------------------------------------------------------

    def initialize(self) -> None:
        """
        Train a single universal embedding using signed losses.
        """
        cfg = self.config
        model = self.model.to(self.device)
        model.train()

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        embed_layer = model.get_input_embeddings()

        # init embedding
        if cfg.rand_init:
            optim_embeds = torch.randn(
                (1, cfg.num_tokens, model.config.hidden_size),
                device=self.device,
                requires_grad=True,
            )
        else:
            ids = self.tokenizer(
                cfg.optim_str_init,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"].to(self.device)
            optim_embeds = embed_layer(ids).detach().clone().requires_grad_(True)

        opt = torch.optim.Adam([optim_embeds], lr=cfg.lr)

        if not self.cb_data and not self.hp_data:
            raise ValueError(
                "UniversalEmbeddingAttack requires training data "
                "(circuit_breakers and/or honeypots)."
            )

        # build training pool
        pool = []
        if cfg.w_circuit_breakers and self.cb_data:
            for ex in self.cb_data:
                pool.append(("cb", ex))
        if cfg.w_honeypots and self.hp_data:
            for ex in self.hp_data:
                pool.append(("hp", ex))

        assert pool, "No training data provided for universal embedding attack."

        for step in tqdm.trange(cfg.num_steps, desc="Training universal embedding"):
            kind, ex = pool[step % len(pool)]
            prompt = ex["prompt"]
            # take response, or output or fallback to target if neither are present
            target = ex.get("response") or ex.get("output") or ex.get("target")

            messages = [{"role": "user", "content": prompt + " {optim_str}"}]
            template = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            before, after = template.split("{optim_str}")

            before_ids = self.tokenizer(
                before, return_tensors="pt", add_special_tokens=False
            )["input_ids"].to(self.device)
            after_ids = self.tokenizer(
                after, return_tensors="pt", add_special_tokens=False
            )["input_ids"].to(self.device)
            target_ids = self.tokenizer(
                target, return_tensors="pt", add_special_tokens=False
            )["input_ids"].to(self.device)

            before_embeds = embed_layer(before_ids)
            after_embeds = embed_layer(after_ids)
            target_embeds = embed_layer(target_ids)

            input_embeds = torch.cat(
                [before_embeds, optim_embeds, after_embeds, target_embeds.detach()],
                dim=1,
            )

            logits = model(inputs_embeds=input_embeds, use_cache=False).logits
            shift = input_embeds.shape[1] - target_ids.shape[1]
            shift_logits = logits[..., shift - 1 : shift - 1 + target_ids.shape[1], :]

            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                target_ids.view(-1),
            )

            # signed objective
            if kind == "cb":
                loss = cfg.w_circuit_breakers * ce
            else:  # honeypot
                loss = cfg.w_honeypots * ce

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % cfg.log_every == 0:
                logging.info(
                    "[universal] step=%d kind=%s ce=%.4f weighted=%.4f",
                    step,
                    kind,
                    ce.item(),
                    loss.item(),
                )

        self.optim_embeds = optim_embeds.detach().cpu()
        model.eval()

    # --------------------------------------------------------
    # Run one example
    # --------------------------------------------------------
    @torch.no_grad()
    def run_example(
        self,
        *,
        behavior_id: str,
        prompt: str,
        target: Optional[str] = None,
        variant_id: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,  # swallow anything else from runner
    ) -> Dict[str, Any]:
        """
        Apply the frozen universal embedding to a single prompt.

        target / variant_id are accepted for API compatibility,
        but are not used.
        """
        assert self.optim_embeds is not None, "Call initialize() first."

        start = time.time()

        messages = [{"role": "user", "content": prompt}]
        template = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        ids = self.tokenizer(
            template, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.device)

        base_embeds = self.model.get_input_embeddings()(ids)
        input_embeds = torch.cat(
            [base_embeds, self.optim_embeds.to(self.device)], dim=1
        )

        output_ids = self.model.generate(
            inputs_embeds=input_embeds,
            max_length=self.max_generation_length,
            use_cache=False,
            do_sample=False,
        )

        gen_text = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()

        return {
            "prompt": prompt,
            "generated": gen_text,
            "attack_metadata": {
                "behavior_id": behavior_id,
                "variant_id": variant_id,
                "duration_seconds": time.time() - start,
                "attack_type": "universal_soft_prompt",
            },
        }