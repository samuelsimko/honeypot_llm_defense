"""
attacks/soft_prompt_attack.py

Soft prompt optimization attack (per-example) with optional prompt/target augmentation.

Design goals:
- Works with either a plain HF model or a defense (HF base + LoRA adapter).
- No judging logic here. This returns generations + metadata; a runner can judge later.
- Clean interface:
    attack = SoftPromptAttack(...)
    attack.initialize()          # optional one-time setup
    out = attack.run_example(...)    # one behavior/example
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


# ----------------------------
# Config + Results
# ----------------------------

@dataclass
class SoftOptConfig:
    num_steps: int = 200
    optim_str_init: str = "x " * 5
    rand_init: bool = False
    num_tokens: int = 5
    lr: float = 0.01
    early_stop_loss: float = 0.001
    add_space_before_target: bool = False
    seed: Optional[int] = None
    verbose: bool = False
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SoftOptConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        init_data = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        if "str_length" in extra and "optim_str_init" not in init_data:
            init_data["optim_str_init"] = "x " * int(extra["str_length"])
        init_data["extra_fields"] = data
        return cls(**init_data)


@dataclass
class SoftOptResult:
    losses: List[float]
    # optimized soft prompt embeddings only
    optim_embeds: torch.Tensor
    # full input embeds used for generation (prefix + optim + suffix), excluding target tokens
    input_embeds: torch.Tensor


# ----------------------------
# Core optimization
# ----------------------------

def _ensure_messages(messages: Union[str, List[dict]]) -> List[dict]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return messages


def run_soft_opt(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    target: str,
    config: SoftOptConfig,
    device: torch.device,
) -> SoftOptResult:
    """
    Optimize a soft prompt slot {optim_str} in the last message to cause the model to generate `target`.
    """
    model.enable_input_require_grads()

    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

    msgs = _ensure_messages(messages)
    if not any("{optim_str}" in d["content"] for d in msgs):
        msgs[-1]["content"] = msgs[-1]["content"] + "{optim_str}"

    template = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # Some tokenizers include BOS in string template; strip it once if present
    if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
        template = template.replace(tokenizer.bos_token, "", 1)

    if "{optim_str}" not in template:
        raise ValueError("Chat template did not preserve {optim_str} placeholder.")

    before_str, after_str = template.split("{optim_str}")

    tgt = (" " + target) if config.add_space_before_target else target

    before_ids = tokenizer(before_str, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    after_ids = tokenizer(after_str, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    target_ids = tokenizer(tgt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    emb = model.get_input_embeddings()
    before_embeds = emb(before_ids)
    after_embeds = emb(after_ids)
    target_embeds = emb(target_ids)

    if not config.rand_init:
        optim_ids = tokenizer(config.optim_str_init, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        optim_embeds = emb(optim_ids).detach().clone().requires_grad_(True)
    else:
        optim_embeds = torch.randn((1, config.num_tokens, model.config.hidden_size), device=device).requires_grad_(True)

    dtype = model.get_input_embeddings().weight.dtype
    optim_embeds = optim_embeds.to(dtype).detach().requires_grad_(True)

    opt = torch.optim.Adam([optim_embeds], lr=config.lr)
    losses: List[float] = []

    # We train on (prefix + optim + suffix + target), but we only generate from (prefix + optim + suffix)
    config.verbose = False
    for step in tqdm.tqdm(range(config.num_steps), disable=not config.verbose):
        opt.zero_grad(set_to_none=True)

        train_embeds = torch.cat([before_embeds, optim_embeds, after_embeds, target_embeds.detach()], dim=1)
        out = model(inputs_embeds=train_embeds, use_cache=False)
        logits = out.logits

        shift = train_embeds.shape[1] - target_ids.shape[1]
        # align so we predict target token t from previous position
        shift_logits = logits[..., shift - 1 : shift - 1 + target_ids.shape[1], :].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            target_ids.view(-1),
        )

        losses.append(float(loss.item()))
        if config.verbose:
            logging.info("[soft_opt] step=%d loss=%.6f", step, loss.item())

        if config.early_stop_loss and loss.item() < config.early_stop_loss:
            break

        loss.backward()
        opt.step()

    gen_input_embeds = torch.cat([before_embeds, optim_embeds.detach(), after_embeds], dim=1)

    return SoftOptResult(
        losses=losses,
        optim_embeds=optim_embeds.detach().cpu(),
        input_embeds=gen_input_embeds.detach().cpu(),
    )


# ----------------------------
# Attack class
# ----------------------------

class SoftPromptAttack:
    """
    Per-example soft prompt attack.

    Runner contract:
      - initialize() is called once per job
      - run_example() is called once per behavior/example
    """

    name = "soft_prompt"

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        *,
        device: Optional[str] = None,
        max_generation_length: int = 1536,
        pick_best: str = "min_final_loss",
        **attack_config,   # NEW
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.attack_config = dict(attack_config)
        self.max_generation_length = int(max_generation_length)
        self.pick_best = pick_best

        if device is None:
            self.device = self.model.device
        else:
            self.device = torch.device(device)

    def initialize(self) -> None:
        # One-time setup. Keep minimal and deterministic.
        self.model.eval()
        # Ensure pad token exists (common for llama-like tokenizers).
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def _generate_from_embeds(self, input_embeds: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        input_embeds: [1, S, D] on self.device and correct dtype.
        Returns (generated_ids, decoded_text).
        """
        gen_ids = self.model.generate(
            inputs_embeds=input_embeds,
            max_length=self.max_generation_length,
            do_sample=False,
            use_cache=False,
        )
        text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        return gen_ids[0], text

    def run_example(
        self,
        *,
        behavior_id: str,
        prompt: str,
        target: str,
        variant_id: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,  # swallow anything else from runner
    ) -> Dict[str, Any]:
        """
        Run the soft prompt attack for a SINGLE (prompt, target) pair.

        Runner owns augmentation and variant selection.
        """
        start = time.time()

        cfg = SoftOptConfig.from_dict({**self.attack_config, "seed": seed})

        try:
            # --------------------------------------------------
            # Optimize soft prompt
            # --------------------------------------------------
            res = run_soft_opt(
                model=self.model,
                tokenizer=self.tokenizer,
                messages=prompt,
                target=target,
                config=cfg,
                device=self.device,
            )

            # --------------------------------------------------
            # Generate from optimized embeddings
            # --------------------------------------------------
            input_embeds = res.input_embeds.to(
                device=self.device,
                dtype=self.model.dtype,
            )

            _, gen_text = self._generate_from_embeds(input_embeds)

            return {
                "prompt": prompt,
                "generated": gen_text,
                "attack_metadata": {
                    "behavior_id": behavior_id,
                    "variant_id": variant_id,
                    "losses": res.losses,
                    "final_loss": res.losses[-1] if res.losses else None,
                    "min_loss": min(res.losses) if res.losses else None,
                    "duration_seconds": time.time() - start,
                },
            }

        except Exception as e:
            logging.exception("SoftPromptAttack failed on %s", behavior_id)
            return {
                "prompt": prompt,
                "generated": "",
                "attack_metadata": {
                    "behavior_id": behavior_id,
                    "variant_id": variant_id,
                    "error": str(e),
                    "duration_seconds": time.time() - start,
                },
            }

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()