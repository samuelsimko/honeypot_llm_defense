#!/usr/bin/env python3
import json
from pathlib import Path

OUT_PATH = "experiments/generated_llama_cb_grpo_sweep.json"
EXPERIMENT_NAME = "llama_cb_grpo_matched_sweep"
OUTPUT_ROOT = "runs/experiments"

# ============================================================
# MODELS
# ============================================================

MODELS = {
    "llama3_8b": {
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct"
    }
}
# ============================================================
# CB SETTINGS
# We sweep 6 different CB initializations by varying:
# - margin_ce
# - w_benign
# - w_breaking
# Each will be combined with nodpo / dpo_weak / dpo_mid
# for 6 x 3 = 18 trained defenses.
# ============================================================
CB_PROFILES = [
    {
        "name": "cb_a",
        "train_args": {
            "margin_ce": 0.0,
            "w_benign": 1.0,
            "w_breaking": 1.0,
        },
    },
    {
        "name": "cb_b",
        "train_args": {
            "margin_ce": 0.0,
            "w_benign": 5,
            "w_breaking": 3.0,
        },
    },
    {
        "name": "cb_c",
        "train_args": {
            "margin_ce": 5.0,
            "w_benign": 1.0,
            "w_breaking": 3.0,
        },
    },
    {
        "name": "cb_d",
        "train_args": {
            "margin_ce": 5.0,
            "w_benign": 2,
            "w_breaking": 0.5,
        },
    },
    {
        "name": "cb_e",
        "train_args": {
            "margin_ce": 10.0,
            "w_benign": 1.0,
            "w_breaking": 3.0,
        },
    },
    {
        "name": "cb_f",
        "train_args": {
            "margin_ce": 10.0,
            "w_benign": 0.5,
            "w_breaking": 2,
        },
    },
]

# ============================================================
# HONEYPOT / DPO REGIMES
# ============================================================

DPO_PROFILES = [
    {
        "name": "nodpo",
        "train_args": {
            "w_dpo_hp_over_harm": 0.0,
        },
    },
    {
        "name": "dpo_weak",
        "train_args": {
            "w_dpo_hp_over_harm": 0.5,
            "dpo_beta": 1.0,
            "dpo_margin": 0.0,
            "dpo_detach_harm": True,
        },
    },
    {
        "name": "dpo_mid",
        "train_args": {
            "w_dpo_hp_over_harm": 1.0,
            "dpo_beta": 2.0,
            "dpo_margin": 1.0,
            "dpo_detach_harm": True,
        },
    },
]

# ============================================================
# DATA
# ============================================================

DEFAULT_DATA = {
    "cb_path": "data/circuit_breakers_train.json",
    "honeypot_path": "data/honeypots_qwen_fixed.jsonl",
}

# ============================================================
# ATTACKS
# Keep only fast GRPO to fit the overnight budget.
# ============================================================

ATTACKS = {
    "grpo_prefill_fast": {
        "attack_name": "grpo_prefill",
        "num_variants": 1,
        "attack_config": [
            {
                "lr": 1e-5,
                "num_steps": 100,
                "rollouts_per_step": 2,
                "max_new_tokens": 150,
                "prefill_max_chars": 24,
                "use_elite_buffer": True,
                "elite_buffer_size": 16,
                "seed_jailbreak_bank": [
                    "wildjailbreaks",
                    "jailbreakv"
                ]
            }
        ]
    }
}

# ============================================================
# BENIGN EVALS
# Only GSM8K, limited to 200 examples.
# ============================================================

BENIGN_EVALS = {
    "gsm8k_200": {
        "tasks": "gsm8k",
        "limit": 200
    }
}

# ============================================================
# GENERATION
# ============================================================

def main():
    defenses = {}
    pipelines = {}

    # --------------------------------------------------------
    # Base model
    # --------------------------------------------------------
    base_id = "base_llama3_8b"
    defenses[base_id] = {
        "script": None,
        "base_model": "llama3_8b",
        "output_subdir": base_id,
        "train_args": {},
    }
    pipelines[base_id] = [
        {
            "stage": "attack",
            "defense": base_id,
            "attacks": list(ATTACKS.keys()),
        },
        {
            "stage": "benign_eval",
            "defense": base_id,
            "benign_eval": "gsm8k_200",
        },
    ]

    # --------------------------------------------------------
    # 18 trained defenses = 6 CB settings x 3 DPO settings
    # --------------------------------------------------------
    for cb_prof in CB_PROFILES:
        for dpo_prof in DPO_PROFILES:
            tag = f"cb_llama3_8b_{cb_prof['name']}_{dpo_prof['name']}"

            train_args = {}
            train_args.update(cb_prof["train_args"])
            train_args.update(dpo_prof["train_args"])

            defenses[tag] = {
                "script": "defenses/honeypot_cb.py",
                "base_model": "llama3_8b",
                "output_subdir": tag,
                "data": DEFAULT_DATA,
                "train_args": train_args,
            }

            pipelines[tag] = [
                {
                    "stage": "train",
                    "defense": tag,
                },
                {
                    "stage": "attack",
                    "defense": tag,
                    "attacks": list(ATTACKS.keys()),
                },
                {
                    "stage": "benign_eval",
                    "defense": tag,
                    "benign_eval": "gsm8k_200",
                },
            ]

    experiment = {
        "meta": {
            "experiment_name": EXPERIMENT_NAME,
            "output_root": OUTPUT_ROOT,
        },
        "cluster": {
            "partition": "tamper_resistance",
            "account": "zhijing_jin",
            "gres": "gpu:1",
            "time_train": "02:00:00",
            "time_attack": "01:30:00",
            "time_benign": "01:00:00",
        },
        "models": MODELS,
        "datasets": {
            "harmbench_csv": "data/harmbench_behaviors_text_val.csv",
            "harmbench_targets": "data/harmbench_targets_text.json",
        },
        "defenses": defenses,
        "attacks": ATTACKS,
        "benign_evals": BENIGN_EVALS,
        "pipelines": pipelines,
        "run_pipelines": list(pipelines.keys()),
    }

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(experiment, f, indent=2)

    print(f"✅ Wrote {OUT_PATH}")
    print(f"Defenses: {len(defenses)}")
    print(f"Pipelines: {len(pipelines)}")
    print("Sweep:")
    print(f"  - CB settings: {len(CB_PROFILES)}")
    print(f"  - DPO regimes: {len(DPO_PROFILES)}")
    print(f"  - Trained defenses: {len(CB_PROFILES) * len(DPO_PROFILES)}")
    print(f"  - Base defenses: 1")
    print(f"  - Total pipelines: {len(pipelines)}")


if __name__ == "__main__":
    main()
