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
import gc

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

    # from attacks.nanogcg_attack import NanoGCGAttack
    # registry["nanogcg"] = NanoGCGAttack

    from attacks.grpo_prefill_attack import GRPOEliteAttack
    registry["grpo_prefill"] = GRPOEliteAttack

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

    parser.add_argument(
        "--behavior-idx",
        type=int,
        default=None,
        help="If set, only run a single behavior by index"
    )

    # Judging
    parser.add_argument("--behavior-label", default="harmful behavior")

    parser.add_argument(
        "--judge-only",
        action="store_true",
        help="Skip attack generation; only run judging + stats from existing completions.pkl",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.lora and not os.path.exists(args.lora):
        raise RuntimeError(f"Expected LoRA adapter at {args.lora}, but it does not exist.")

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
        print("Limited to %d rows", args.limit)
        print(df.head())

    targets_map = json.load(open(args.harmbench_targets))
    attack_configs = load_json_or_jsonl(args.attack_config_path)

    if args.behavior_idx is not None:
        if args.behavior_idx < 0 or args.behavior_idx >= len(df):
            raise ValueError("behavior-idx out of range")
        df = df.iloc[[args.behavior_idx]].reset_index(drop=True)

    LOGGER.info("Loaded %d attack configs", len(attack_configs))
    LOGGER.info("Loaded %d benchmark rows", len(df))


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    model = None
    tokenizer = None

    if not args.judge_only:
        model, tokenizer = load_model_and_tokenizer(
            args.model,
            lora_path=args.lora,
            device=args.device,
            torch_dtype=args.torch_dtype,
        )
        print("Loaded model with lora: ", args.lora)

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------
    if not args.judge_only:
        completions = load_pickle(completions_path) if os.path.exists(completions_path) else []

        AttackCls = get_attack_class(args.attack)

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

            for i, row in enumerate(
                tqdm.tqdm(df.itertuples(), total=len(df), desc="Behaviors", leave=False)
            ):
                bid = row.BehaviorID
                prompt = row.Behavior
                target = targets_map.get(bid)
                if target is None:
                    continue

                # --------------------------------------------------
                # GRPO-style attacks: SINGLE CALL, MULTIPLE VARIANTS
                # --------------------------------------------------
                if args.attack == "grpo_prefill":
                    out = attack.run_example(
                        prompt=prompt,
                        behavior_id=bid,
                        target=target,
                        variant_id=0,
                    )

                    meta = out.get("attack_metadata", {})
                    variants = meta.get("successful_jailbreaks", [])

                    # Fallback: ensure at least one variant
                    if not variants:
                        variants = [{
                            "target_reply": out["generated"]
                        }]

                    for v_id, v in enumerate(variants[:args.num_variants]):
                        completions.append({
                            "attack": args.attack,
                            "attack_config_id": cfg_id,
                            "attack_config": cfg,
                            "behavior_id": bid,
                            "variant_id": v_id,
                            "prompt": prompt,
                            "target": target,
                            "generated": v["target_reply"],
                            "attack_metadata": meta,
                        })

                # --------------------------------------------------
                # Non-GRPO attacks (soft prompt, embedding, etc.)
                # --------------------------------------------------
                else:
                    for v_id in range(args.num_variants):
                        out = attack.run_example(
                            prompt=prompt,
                            behavior_id=bid,
                            target=target,
                            variant_id=v_id,
                        )

                        out = normalize_attack_output(out, prompt)

                        completions.append({
                            "attack": args.attack,
                            "attack_config_id": cfg_id,
                            "attack_config": cfg,
                            "behavior_id": bid,
                            "variant_id": v_id,
                            "prompt": out["prompt"],
                            "target": target,
                            "generated": out["generated"],
                            "attack_metadata": out["attack_metadata"],
                        })

                if (i + 1) % args.save_every == 0:
                    save_pickle(completions, completions_path)

            save_pickle(completions, completions_path)

        LOGGER.info("Saved %d completions", len(completions))

        del attack
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
    else:
        if not os.path.exists(completions_path):
            raise FileNotFoundError(
                f"--judge-only specified but completions.pkl not found in {args.output_dir}"
            )

        completions = load_pickle(completions_path)
        LOGGER.info("Loaded %d existing completions (judge-only mode)", len(completions))

    # Judging (BATCHED, SAFE)
    # --------------------------------------------------------

    torch.cuda.empty_cache()
    gc.collect()

    prompts = [c["prompt"] for c in completions]
    responses = [c["generated"] for c in completions]

    raw_results = judge_sequence(
        prompts,
        responses,
        behavior=args.behavior_label,
    )

    judged = []
    for c, j in zip(completions, raw_results):
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
