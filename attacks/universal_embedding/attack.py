"""
Soft Prompt Optimization using Circuit Breakers for Training
and HarmBench for Evaluation.

Process:
  1. Load Circuit Breakers dataset (refusal + harmful examples)
  2. Learn a universal soft prompt embedding that minimizes loss on all CB data
  3. Freeze this embedding
  4. Test on HarmBench dataset using HarmBench and StrongReject evaluators
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import gc
from judges.judges import judge_sequence
from judges.judges import normalize_judge_result


import os
import sys
import json
import time
import torch
import tqdm
import logging
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from attacks.behavior_targets.augment import generate_prompt_target_variants

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
@dataclass
class CBTrainConfig:
    lr: float = 5e-3
    num_steps: int = 2000
    num_tokens: int = 2
    optim_str_init: str = "x"*2
    rand_init: bool = False
    seed: int = 42
    device: str = "cuda:0"
    log_every: int = 50
    save_path: str = "cb_trained_embedding.pt"


# ------------------------------------------------------------
# Load Circuit Breakers
# ------------------------------------------------------------

def load_json_or_jsonl(path: str):
    with open(path, "r") as f:
        if path.endswith(".json"):
            return json.load(f)
        elif path.endswith(".jsonl"):
            return [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported file extension: {path}")


def load_circuit_breakers(path: str, limit: int = None):
    """Loads Circuit Breakers JSON/JSONL into lists of (prompt, refusal, harmful)."""
    data = load_json_or_jsonl(path)
    if limit:
        data = data[:limit]
    prompts = [d["prompt"] for d in data if "prompt" in d]
    refusals = [(d.get("llama3_output") or "I'm sorry, I cannot do that.") for d in data]
    harmful = [d.get("output", "") for d in data]
    # only get first 100 characters
    # refusals = [r[:100] for r in refusals]
    # harmful = [h[:100] for h in harmful]

    return prompts, refusals, harmful


# ------------------------------------------------------------
# Train universal embedding
# ------------------------------------------------------------
def train_cb_embedding(model, tokenizer, prompts, targets, config: CBTrainConfig):
    model.to(config.device)
    model.train()
    embedding_layer = model.get_input_embeddings()

    if config.rand_init:
        optim_embeds = torch.randn((1, config.num_tokens, model.config.hidden_size),
                                   device=config.device, requires_grad=True)
    else:
        optim_ids = tokenizer(config.optim_str_init, return_tensors="pt", add_special_tokens=False)["input_ids"].to(config.device)
        optim_embeds = embedding_layer(optim_ids).detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([optim_embeds], lr=config.lr)
    losses = []

    for step in tqdm.trange(config.num_steps, desc="Training universal soft prompt"):
        prompt, target = prompts[step], targets[step]
        messages = [{"role": "user", "content": prompt + " {optim_str}"}]
        template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        before_str, after_str = template.split("{optim_str}")

        before_ids = tokenizer(before_str, return_tensors="pt", add_special_tokens=False)["input_ids"].to(config.device)
        after_ids = tokenizer(after_str, return_tensors="pt", add_special_tokens=False)["input_ids"].to(config.device)
        target_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False)["input_ids"].to(config.device)

        before_embeds = embedding_layer(before_ids)
        after_embeds = embedding_layer(after_ids)
        target_embeds = embedding_layer(target_ids)

        input_embeds = torch.cat([before_embeds, optim_embeds, after_embeds, target_embeds.detach()], dim=1)
        outputs = model(inputs_embeds=input_embeds, use_cache=False)
        logits = outputs.logits

        shift = input_embeds.shape[1] - target_ids.shape[1]
        shift_logits = logits[..., shift - 1:shift - 1 + target_ids.shape[1], :]
        shift_labels = target_ids
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % config.log_every == 0:
            print(f"[Step: {step}] Mean of last {config.log_every} losses = {np.mean(losses[-config.log_every:]):.4f}")

    torch.save(optim_embeds.detach().cpu(), config.save_path)
    print(f"✅ Saved universal embedding to {config.save_path}")
    return optim_embeds.detach().cpu(), losses


def generate_universal_completions(
    model,
    tokenizer,
    optim_embeds,
    harmbench_csv,
    harmbench_targets,
    device,
    max_length=512,
    limit=None,
):
    df = pd.read_csv(harmbench_csv)
    df = df[df["FunctionalCategory"] == "standard"].reset_index(drop=True)
    targets_map = json.load(open(harmbench_targets))

    completions = []
    model.eval()

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        if limit is not None and len(completions) >= limit:
            break
        behavior_id = row["BehaviorID"]
        prompt = row["Behavior"]
        target = targets_map.get(behavior_id)
        if target is None:
            continue

        messages = [{"role": "user", "content": prompt}]
        template = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        input_ids = tokenizer(
            template,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(device)

        before_embeds = model.get_input_embeddings()(input_ids)
        input_embeds = torch.cat(
            [before_embeds, optim_embeds.to(device)], dim=1
        )

        with torch.no_grad():
            output = model.generate(
                inputs_embeds=input_embeds,
                max_length=max_length,
                use_cache=False,
            )

        gen_text = tokenizer.decode(output[0], skip_special_tokens=True)

        completions.append({
            "behavior_id": behavior_id,
            "prompt": prompt,
            "target": target,
            "generated": gen_text,
        })

    return completions

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--cb-path", type=str, required=True)
    parser.add_argument("--benchmark-data-csv", type=str, required=True)
    parser.add_argument("--benchmark-targets-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="cb_softopt_eval")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-completions", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = CBTrainConfig(device=args.device, save_path=os.path.join(args.output_dir, "cb_embedding.pt"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map=args.device)

    # 1️⃣ Load circuit breakers
    prompts, refusals, harmful = load_circuit_breakers(args.cb_path, args.limit)
    all_prompts = prompts
    all_targets = harmful

    # 2️⃣ Train one embedding
    optim_embeds, losses = train_cb_embedding(model, tokenizer, all_prompts, all_targets, config)

    completions = generate_universal_completions(
        model=model,
        tokenizer=tokenizer,
        optim_embeds=optim_embeds,
        harmbench_csv=args.benchmark_data_csv,
        harmbench_targets=args.benchmark_targets_json,
        device=args.device,
        limit=args.limit_completions,
    )

    # remove model from memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    with open(os.path.join(args.output_dir, "completions.pkl"), "wb") as f:
        pickle.dump(completions, f)
    print(f"✅ Saved completions to {args.output_dir}/completions.pkl")

    results = judge_sequence(
        [c["prompt"] for c in completions],
        [c["generated"] for c in completions],
        behavior="harmful behavior",
    )
    print(results)

    # normalize results
    normalized_results = [normalize_judge_result(r) for r in results]
    with open(os.path.join(args.output_dir, "normalized_results.pkl"), "wb") as f:
        pickle.dump(normalized_results, f)
    print(f"✅ Saved normalized results to {args.output_dir}/normalized_results.pkl")
    print(normalized_results)

if __name__ == "__main__":
    main()