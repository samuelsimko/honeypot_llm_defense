#!/usr/bin/env python3
"""
run_attack.py

Unified attack runner using Attack objects (attacks/base.py).
"""

import argparse
import json
import os
import pickle
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import tqdm
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from judges.judges import judge_sequence, normalize_judge_result

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("attack")


# ============================================================
# IO helpers
# ============================================================

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)

        if head in ("[", "{"):
            try:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                f.seek(0)

        for line in f:
            s = line.strip()
            if s:
                records.append(json.loads(s))

    return records


def save_pickle(obj: Any, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# Model loading
# ============================================================

def load_model_and_tokenizer(
    model_path: str,
    *,
    lora_path: Optional[str],
    device: str,
    torch_dtype: str = "bfloat16",
):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    td = dtype_map.get(torch_dtype.lower(), torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=td,
        device_map="auto" if device.startswith("cuda") else {"": "cpu"},
    )

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    return model, tokenizer


# ============================================================
# Attack registry
# ============================================================

def get_attack_class(attack_name: str):
    registry = {}

    from attacks.soft_prompt_attack import SoftPromptAttack
    registry["soft_prompt"] = SoftPromptAttack

    from attacks.universal_embedding_attack import UniversalEmbeddingAttack
    registry["universal_soft_prompt"] = UniversalEmbeddingAttack

    if attack_name not in registry:
        raise ValueError(
            f"Unknown attack '{attack_name}'. Available: {list(registry.keys())}"
        )

    return registry[attack_name]


# ============================================================
# Output normalization
# ============================================================

def normalize_attack_output(out: Dict[str, Any], fallback_prompt: str) -> Dict[str, Any]:
    prompt = out.get("prompt", fallback_prompt)
    generated = out.get("generated")

    if generated is None:
        raise ValueError(f"Attack output missing 'generated': {out.keys()}")

    meta = out.get("attack_metadata", {})
    if not isinstance(meta, dict):
        meta = {"value": meta}

    return {
        "prompt": prompt,
        "generated": generated,
        "attack_metadata": meta,
    }


# ============================================================
# Training data loading for universal attacks
# ============================================================

def load_training_data(spec: Dict[str, str]) -> Dict[str, Any]:
    """
    Load training datasets for attacks from paths specified in config.

    Expected format:
      {
        "circuit_breakers": "path/to/cb.json",
        "honeypots": "path/to/hp.jsonl"
      }
    """
    data: Dict[str, Any] = {}

    if not spec:
        return data

    if "circuit_breakers" in spec:
        data["circuit_breakers"] = load_json_or_jsonl(spec["circuit_breakers"])

    if "honeypots" in spec:
        data["honeypots"] = load_json_or_jsonl(spec["honeypots"])

    return data

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser("Unified Attack Runner")

    # Core
    parser.add_argument("--attack", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora", default=None)

    # Benchmark
    parser.add_argument("--harmbench-csv", required=True)
    parser.add_argument("--harmbench-targets", required=True)

    # Attack configs
    parser.add_argument("--attack-config-path", required=True)

    # Output
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-every", type=int, default=10)

    # Runtime
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--limit", type=int)

    # Variants
    parser.add_argument("--num-variants", type=int, default=1)

    # Judging
    parser.add_argument("--behavior-label", default="harmful behavior")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    completions_path = os.path.join(args.output_dir, "completions.pkl")
    judged_path = os.path.join(args.output_dir, "judged.pkl")
    stats_path = os.path.join(args.output_dir, f"{args.attack}_stats.json")

    # --------------------------------------------------------
    # Load benchmark
    # --------------------------------------------------------
    df = pd.read_csv(args.harmbench_csv)
    df = df[df["FunctionalCategory"] == "standard"].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    targets_map = json.load(open(args.harmbench_targets))
    attack_configs = load_json_or_jsonl(args.attack_config_path)

    LOGGER.info("Loaded %d attack configs", len(attack_configs))
    LOGGER.info("Loaded %d benchmark rows", len(df))

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        lora_path=args.lora,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------
    completions = load_pickle(completions_path) if os.path.exists(completions_path) else []
    done_keys = {(c["attack_config_id"], c["behavior_id"], c["variant_id"]) for c in completions}

    AttackCls = get_attack_class(args.attack)
    init_prompts = [r.Behavior for r in df.itertuples()]

    for cfg_id, cfg in enumerate(tqdm.tqdm(attack_configs, desc="Attack configs")):
        training_data = load_training_data(cfg.get("training_data", {}))
        hyper = cfg.get("hyperparams", cfg)

        attack = AttackCls(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            **training_data,
            **hyper,
        )

        attack.initialize()

        for i, row in enumerate(tqdm.tqdm(df.itertuples(), total=len(df), desc="Behaviors", leave=False)):
            bid = row.BehaviorID
            prompt = row.Behavior
            target = targets_map.get(bid)
            if target is None:
                continue

            variants = [(prompt, target)] if args.num_variants == 1 else [
                (prompt, target) for _ in range(args.num_variants)
            ]

            for v_id, (p, t) in enumerate(variants):
                key = (cfg_id, bid, v_id)
                if key in done_keys:
                    continue

                out = attack.run_example(
                    prompt=p,
                    behavior_id=bid,
                    target=t,
                    variant_id=v_id,
                )
                out = normalize_attack_output(out, p)

                completions.append({
                    "attack": args.attack,
                    "attack_config_id": cfg_id,
                    "attack_config": cfg,
                    "behavior_id": bid,
                    "variant_id": v_id,
                    "prompt": out["prompt"],
                    "target": t,
                    "generated": out["generated"],
                    "attack_metadata": out["attack_metadata"],
                })
                done_keys.add(key)

            if (i + 1) % args.save_every == 0:
                save_pickle(completions, completions_path)

        save_pickle(completions, completions_path)

    LOGGER.info("Saved %d completions", len(completions))

    # --------------------------------------------------------
    # Judging
    # --------------------------------------------------------
    judged = []
    for c in tqdm.tqdm(completions, desc="Judging"):
        j = judge_sequence([c["prompt"]], [c["generated"]], behavior=args.behavior_label)[0]
        c = dict(c)
        c["judge_raw"] = j
        c["judge_norm"] = normalize_judge_result(j)
        judged.append(c)

    save_pickle(judged, judged_path)

    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------
    variant_scores = defaultdict(list)
    behavior_scores = defaultdict(lambda: defaultdict(list))

    for r in judged:
        bid = r["behavior_id"]
        for k, v in r["judge_norm"].items():
            variant_scores[k].append(v)
            behavior_scores[bid][k].append(v)

    stats = {}
    for k, v in variant_scores.items():
        stats[f"variants_mean_{k}"] = float(np.mean(v)) if v else float("nan")
        stats[f"behavior_max_mean_{k}"] = float(np.mean([max(behavior_scores[b][k]) for b in behavior_scores]))

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    LOGGER.info("Attack run complete")
    LOGGER.info(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()