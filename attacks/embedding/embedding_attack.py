"""
Soft Prompt Optimization Benchmark with Augmented Prompt-Target Variants
and HarmBench + StrongReject evaluation.

This version:
 - Uses no KV cache (compatible with all Transformers >=4.30)
 - Imports augmentation utilities from attacks/behavior_targets/augment.py
 - Runs multiple prompt-target augmentations per harmful behavior
 - Evaluates results with HarmBench and StrongReject classifiers
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import json
import os
import tqdm
import pickle
import logging
import time
import gc
from dataclasses import dataclass, field
from typing import List, Union, Dict, Any

import torch
import numpy as np
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset
from peft import PeftModel, PeftConfig

# 🔹 NEW: Augmentation module import
from attacks.behavior_targets.augment import generate_prompt_target_variants


# ------------------------------------------------------------
# Soft Prompt Optimization Config & Result
# ------------------------------------------------------------
@dataclass
class SoftOptConfig:
    """Configuration for the soft prompt optimization process."""
    num_steps: int = 100
    optim_str_init: str = "x " * 10
    rand_init: bool = False
    num_tokens: int = 10
    lr: float = 0.01
    early_stop_loss: float = None
    add_space_before_target: bool = False
    device: str = "cuda:0"
    seed: int = None
    verbose: bool = False
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        init_data = {k: v for k, v in data.items() if k in known_keys}
        extra_data = {k: v for k, v in data.items() if k not in known_keys}
        if "str_length" in extra_data and "optim_str_init" not in init_data:
            init_data["optim_str_init"] = "x " * extra_data["str_length"]
        init_data["extra_fields"] = data
        return cls(**init_data)


@dataclass
class SoftOptResult:
    """Result of the soft prompt optimization."""
    losses: List[float]
    optim_embeds: torch.Tensor
    input_embeds: torch.Tensor


# ------------------------------------------------------------
# Core Soft Prompt Optimization (NO KV caching)
# ------------------------------------------------------------
def run_soft_opt(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    target: str,
    config: SoftOptConfig = None,
):
    """Optimizes a soft prompt to make the model generate `target` given `messages`."""
    model.enable_input_require_grads()
    if config is None:
        config = SoftOptConfig()
    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not any("{optim_str}" in d["content"] for d in messages):
        messages[-1]["content"] += "{optim_str}"

    model = model.to(config.device)
    template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
        template = template.replace(tokenizer.bos_token, "", 1)
    before_str, after_str = template.split("{optim_str}")
    target = " " + target if config.add_space_before_target else target

    before_ids = tokenizer(before_str, return_tensors="pt", add_special_tokens=False)["input_ids"].to(config.device)
    after_ids = tokenizer(after_str, return_tensors="pt", add_special_tokens=False)["input_ids"].to(config.device)
    target_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False)["input_ids"].to(config.device)

    embedding_layer = model.get_input_embeddings()
    before_embeds = embedding_layer(before_ids)
    after_embeds = embedding_layer(after_ids)
    target_embeds = embedding_layer(target_ids)

    if not config.rand_init:
        optim_ids = tokenizer(config.optim_str_init, return_tensors="pt", add_special_tokens=False)["input_ids"].to(config.device)
        optim_embeds = embedding_layer(optim_ids).detach().clone().requires_grad_(True)
    else:
        optim_embeds = torch.randn((1, config.num_tokens, model.config.hidden_size), device=config.device).requires_grad_(True)

    optimizer = torch.optim.Adam([optim_embeds], lr=config.lr)
    losses = []

    for step in tqdm.tqdm(range(config.num_steps)):
        optimizer.zero_grad()
        input_embeds = torch.cat([before_embeds, optim_embeds, after_embeds, target_embeds.detach()], dim=1)
        outputs = model(inputs_embeds=input_embeds, use_cache=False)
        logits = outputs.logits

        shift = input_embeds.shape[1] - target_ids.shape[1]
        shift_logits = logits[..., shift - 1:shift - 1 + target_ids.shape[1], :].contiguous()
        shift_labels = target_ids
        loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        losses.append(loss.item())
        if config.verbose:
            print(f"[Step {step}] loss = {loss.item():.4f}")
        if config.early_stop_loss and loss.item() < config.early_stop_loss:
            break
        # loss.backward()
        loss.backward(retain_graph=True)
        optimizer.step()

    return SoftOptResult(
        losses=losses,
        optim_embeds=optim_embeds.detach().cpu(),
        input_embeds=torch.cat([before_embeds, optim_embeds, after_embeds], dim=1).cpu(),
    )


# ------------------------------------------------------------
# Benchmark Runner (with Augmentation + Judging)
# ------------------------------------------------------------
def run_single_benchmark(
    target_model: PreTrainedModel,
    target_tokenizer: PreTrainedTokenizer,
    prompt: str,
    target: str,
    attack_config_params: dict,
    behavior_id: str,
    max_generation_length: int = 512,
):
    """Runs soft-opt with multiple augmented variants and judges the results."""
    start_time = time.time()
    device = target_model.device
    soft_opt_config = SoftOptConfig.from_dict({**attack_config_params, "device": str(device)})

    output_data = {
        "behavior_id": behavior_id,
        "prompt": prompt,
        "target": target,
        "attack_config": attack_config_params,
        "variants": [],
        "status": "failed",
        "duration_seconds": 0,
    }

    try:
        variants = generate_prompt_target_variants(prompt, target, n=3, seed=soft_opt_config.seed)
        logging.info(f"Generated {len(variants)} augmented variants for {behavior_id}")

        for v_idx, (aug_prompt, aug_target) in enumerate(variants):
            logging.info(f"  ▶ Running Variant {v_idx+1}/{len(variants)}")

            soft_opt_result = run_soft_opt(
                model=target_model,
                tokenizer=target_tokenizer,
                messages=aug_prompt,
                target=aug_target,
                config=soft_opt_config,
            )

            input_embeds_for_gen = soft_opt_result.input_embeds.to(device=device, dtype=target_model.dtype)
            with torch.no_grad():
                gen_output = target_model.generate(
                    inputs_embeds=input_embeds_for_gen,
                    max_length=max_generation_length,
                    output_hidden_states=False,
                )
            generated_str = target_tokenizer.decode(gen_output[0], skip_special_tokens=True).strip()

            variant_entry = {
                "variant_index": v_idx,
                "aug_prompt": aug_prompt,
                "aug_target": aug_target,
                "generated_output": generated_str,
                "losses": soft_opt_result.losses,
            }

            output_data["variants"].append(variant_entry)
            # print stuff
            print(f"Prompt: {aug_prompt}")
            print(f"Generated string: {generated_str}")

        output_data["status"] = "success"

    except Exception as e:
        logging.error(f"Failed on {behavior_id}: {e}", exc_info=True)
        output_data["status"] = "failed"
        output_data["error_message"] = str(e)

    finally:
        output_data["duration_seconds"] = time.time() - start_time
        torch.cuda.empty_cache()

    return output_data