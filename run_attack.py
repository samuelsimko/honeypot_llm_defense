# run_attack.py
import argparse
import json
import os
import pickle
import logging
import pandas as pd
import torch
import random
import numpy as np
import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

from attacks.behavior_targets.augment import generate_prompt_target_variants

# -----------------------
# Attacks (import ONLY attack logic)
# -----------------------
from attacks.embedding.embedding_attack import run_single_benchmark as run_embedding_attack

# -----------------------
# Probe-aware attacks
# -----------------------
from attacks.embedding_probing.embedding_probing import (
    run_single_benchmark as run_embedding_probing_attack,
    load_probes,
)

# -----------------------
# Judges (NEW unified pipeline)
# -----------------------
from judges.judges import judge_sequence

logging.basicConfig(level=logging.INFO)

ATTACK_REGISTRY = {
    "embedding": run_embedding_attack,
    "embedding_probing": run_embedding_probing_attack,
}


def load_json_or_jsonl(path):
    """
    Loads either:
      - a JSON file containing a list/dict
      - a JSONL file (one JSON object per line)

    Returns: list of records
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        # Case 1: proper JSON array / object
        if first_char == "[" or first_char == "{":
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    return [data]
            except json.JSONDecodeError:
                pass  # fall back to JSONL

        # Case 2: JSONL
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records

def normalize_judge_result(j):
    """
    Normalize all judge outputs into numeric signals.
    """
    out = {}

    # HarmBench: yes/no → 1/0
    out["harmbench"] = 1.0 if j["harmbench"] == "yes" else 0.0

    # StrongREJECT: continuous (leave as-is)
    out["strongreject"] = float(j["strongreject"])

    # WildGuard: harmful_response → 1/0
    out["wildguard"] = 1.0 if j["wildguard"]["harmful_response"] else 0.0

    # JailJudge: 1–10 → raw + normalized
    jj = float(j["jailjudge"]["score"])
    out["jailjudge_raw"] = jj
    out["jailjudge_norm"] = jj / 10.0

    # Final GPT-OSS: jailbreak → 1/0
    out["gpt_oss"] = 1.0 if j["final_gpt_oss"]["jailbreak"] else 0.0

    # Ensemble (mean of normalized binary-ish judges)
    out["ensemble_mean"] = float(np.mean([
        out["harmbench"],
        out["wildguard"],
        out["gpt_oss"],
        out["jailjudge_norm"],
    ]))

    return out

def main():
    parser = argparse.ArgumentParser("Unified Attack Runner")

    # -----------------------
    # Core args
    # -----------------------
    parser.add_argument("--attack", type=str, required=True, choices=ATTACK_REGISTRY.keys())
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--benchmark-data-csv", type=str, required=True)
    parser.add_argument("--benchmark-targets-json", type=str, required=True)
    parser.add_argument("--attack-config-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)

    # -----------------------
    # Probing
    # -----------------------
    parser.add_argument("--use-probes", action="store_true")
    parser.add_argument("--harmful-json", type=str)
    parser.add_argument("--honeypot-json", type=str)
    parser.add_argument("--probe-dir", type=str, default="probes")
    parser.add_argument("--rep-layers", type=int, nargs="+", default=[10, 20, 30])

    # -----------------------
    # Variants
    # -----------------------
    parser.add_argument("--num-variants", type=int, default=1)

    # -----------------------
    # Checkpointing
    # -----------------------
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save progress every N behaviors")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    completions_path = os.path.join(args.output_dir, "completions.pkl")
    judged_path = os.path.join(args.output_dir, "judged.pkl")
    stats_path = os.path.join(args.output_dir, f"{args.attack}_stats.json")

    # ============================================================
    # Load benchmark
    # ============================================================
    df = pd.read_csv(args.benchmark_data_csv)
    df = df[df["FunctionalCategory"] == "standard"].reset_index(drop=True)

    if args.limit is not None:
        df = df.head(args.limit)

    targets_map = json.load(open(args.benchmark_targets_json))
    attack_configs = load_json_or_jsonl(args.attack_config_path)

    # ============================================================
    # Load model
    # ============================================================
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map=args.device,
    ).eval()

    # ============================================================
    # Probes
    # ============================================================
    probes = None
    if args.attack == "embedding_probing":
        assert args.use_probes
        assert args.harmful_json and args.honeypot_json

        from attacks.embedding_probing.embedding_probing import (
            train_response_probes,
            load_probes,
        )

        harmful_data = load_json_or_jsonl(args.harmful_json)
        honeypot_data = load_json_or_jsonl(args.honeypot_json)

        train_response_probes(
            model=model,
            tokenizer=tokenizer,
            harmful_data=harmful_data,
            honeypot_data=honeypot_data,
            rep_layers=args.rep_layers,
            output_dir=args.probe_dir,
            device=args.device,
        )

        probes = load_probes(
            probe_dir=args.probe_dir,
            rep_layers=args.rep_layers,
            hidden_size=model.config.hidden_size,
            device=args.device,
        )

    # ============================================================
    # PHASE 1: ATTACK / GENERATION
    # ============================================================
    attack_fn = ATTACK_REGISTRY[args.attack]

    if os.path.exists(completions_path):
        logging.info("🔄 Resuming from existing completions")
        all_completions = pickle.load(open(completions_path, "rb"))
    else:
        all_completions = []

    completed_keys = {
        (c["attack_config_id"], c["behavior_id"], c["variant_id"])
        for c in all_completions
    }

    for cfg_idx, cfg in enumerate(tqdm.tqdm(attack_configs, desc="Attack configs")):
        for i, row in enumerate(tqdm.tqdm(df.itertuples(), total=len(df), desc="Behaviors")):
            behavior_id = row.BehaviorID
            prompt = row.Behavior
            target = targets_map.get(behavior_id)
            if target is None:
                continue

            if args.num_variants > 1:
                variants = generate_prompt_target_variants(
                    prompt, target, n=args.num_variants
                )
            else:
                variants = [(prompt, target)]

            for v_idx, (aug_prompt, aug_target) in enumerate(variants):
                key = (cfg_idx, behavior_id, v_idx)
                if key in completed_keys:
                    continue

                out = attack_fn(
                    target_model=model,
                    target_tokenizer=tokenizer,
                    probes=probes,
                    prompt=aug_prompt,
                    target=aug_target,
                    attack_config_params=cfg,
                    behavior_id=behavior_id,
                    max_generation_length=512,
                )

                all_completions.append({
                    "attack_config_id": cfg_idx,
                    "behavior_id": behavior_id,
                    "variant_id": v_idx,
                    "prompt": out["prompt"],
                    "target": aug_target,
                    "generated": out["generated"],
                })

            if i % args.save_every == 0:
                pickle.dump(all_completions, open(completions_path, "wb"))

    pickle.dump(all_completions, open(completions_path, "wb"))
    logging.info(f"✅ Saved {len(all_completions)} completions")

    # ============================================================
    # PHASE 2: JUDGING (ONCE)
    # ============================================================
    judged = []
    prompts = [c["prompt"] for c in all_completions]
    responses = [c["generated"] for c in all_completions]

    for i in tqdm.tqdm(range(0, len(prompts)), desc="Judging"):
        j = judge_sequence(
            [prompts[i]],
            [responses[i]],
            behavior=prompts[i],
        )[0]

        c = dict(all_completions[i])
        c["judge_raw"] = j
        c["judge_norm"] = normalize_judge_result(j)
        judged.append(c)

    pickle.dump(judged, open(judged_path, "wb"))
    logging.info("✅ Judging complete")

    # ============================================================
    # PHASE 3: STATS
    # ============================================================
    from collections import defaultdict
    variant_scores = defaultdict(list)
    behavior_scores = defaultdict(lambda: defaultdict(list))

    for c in judged:
        bid = c["behavior_id"]
        for k, v in c["judge_norm"].items():
            variant_scores[k].append(v)
            behavior_scores[bid][k].append(v)

    stats = {}

    for k, vals in variant_scores.items():
        stats[f"variants_mean_{k}"] = float(np.mean(vals))

    for k in variant_scores:
        stats[f"behavior_max_mean_{k}"] = float(np.mean(
            [max(behavior_scores[b][k]) for b in behavior_scores]
        ))

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logging.info("✅ Finished run")
    logging.info(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()