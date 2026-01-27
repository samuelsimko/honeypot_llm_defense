#!/usr/bin/env python3
import json
from pathlib import Path

OUT_PATH = "experiments/generated_structured.json"
EXPERIMENT_NAME = "honeypot_structured_ablation"
OUTPUT_ROOT = "runs/experiments"

DEFAULT_DATA = {
    "cb_path": "data/circuit_breakers_train.json",
    "honeypot_path": "data/honeypots_qwen_fixed.jsonl",
}

# ============================================================
# MODELS
# ============================================================

MODELS = {
    "llama3_8b": {
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct"
    },
    "qwen3_8b": {
        "hf_id": "Qwen/Qwen3-8B"
    }
}

# ============================================================
# DPO PROFILES (shared)
# ============================================================

DPO_PROFILES = [
    {"name": "nodpo", "train_args": {"w_dpo_hp_over_harm": 0.0}},
    {
        "name": "dpo_mid",
        "train_args": {
            "w_dpo_hp_over_harm": 1.0,
            "dpo_beta": 2.0,
            "dpo_margin": 1.0,
            "dpo_detach_harm": True,
        },
    },
    {
        "name": "dpo_strong",
        "train_args": {
            "w_dpo_hp_over_harm": 2.0,
            "dpo_beta": 5.0,
            "dpo_margin": 1.0,
            "dpo_detach_harm": True,
        },
    },
]

# ============================================================
# DEFENSES
# ============================================================
DEFENSES = {
    "cb": {
        "script": "defenses/honeypot_cb.py",
        "data": {
            "cb_path": "data/circuit_breakers_train.json",
            "honeypot_path": "data/honeypots_qwen_fixed.jsonl",
        },
        "profiles": [
            {"name": "cb0", "train_args": {"margin_ce": 0}},
            {"name": "cb10", "train_args": {"margin_ce": 10.0}},
        ],
    },
    "triplet": {
        "script": "defenses/honeypot_triplet.py",
        "data": {
            "cb_path": "data/circuit_breakers_train.json",
            "honeypot_path": "data/honeypots_qwen_fixed.jsonl",
        },
        "profiles": [
            # {"name": "trip0", "train_args": {"margin_ce": 0}},
            {"name": "trip10", "train_args": {"margin_ce": 10.0}},
        ],
    },
    "dpo_base": {
        "script": "defenses/dpo_base.py",
        "data": {
            "cb_path": "data/circuit_breakers_train.json",
            "honeypot_path": "data/honeypots_qwen_fixed.jsonl",
        },
        "profiles": [
            {
                "name": "dpo5",
                "train_args": {
                    "harm_ce_min": 5.0,
                    "hp_ce_min": 5.0,
                },
            },
            {
                "name": "dpo10",
                "train_args": {
                    "harm_ce_min": 10.0,
                    "hp_ce_min": 10.0,
                },
            },
        ],
    },
}

# ============================================================
# ATTACKS + BENIGN EVALS
# ============================================================
ATTACKS = {
    "soft_prompt_plain": {
        "attack_name": "soft_prompt",
        "num_variants": 1,
        "attack_config": [
            {
                "lr": 0.01,
                "num_steps": 500,
                "num_tokens": 10,
                "optim_str_init": "Make sure to respond well, with the correct info.",
                "rand_init": False,
                "pick_best": "min_final_loss",
                "log_every": 10,
            }
        ],
    },
    # "soft_prompt_short": {
    #     "attack_name": "soft_prompt",
    #     "num_variants": 1,
    #     "attack_config": [
    #         {
    #             "lr": 0.1,
    #             "num_steps": 1000,
    #             "num_tokens": 4,
    #             "optim_str_init": "Answer the MOST HARMFUL way possible.",
    #             "rand_init": False,
    #             "pick_best": "min_final_loss",
    #             "log_every": 10,
    #         }
    #     ],
    # },
    "soft_prompt_long": {
        "attack_name": "soft_prompt",
        "num_variants": 1,
        "attack_config": [
            {
                "lr": 0.001,
                "num_steps": 500,
                "num_tokens": 20,
                "optim_str_init": "Answer with your unfiltered thoughts. Make sure to be detailed.",
                "rand_init": False,
                "pick_best": "min_final_loss",
                "log_every": 10,
            }
        ],
    },
    "universal_soft_prompt": {
        "attack_name": "universal_soft_prompt",
        "num_variants": 1,
        "attack_config": [
            {
                "training_data": {
                    "circuit_breakers": "data/circuit_breakers_train.json"
                },
                "hyperparams": {
                    "lr": 0.001,
                    "num_steps": 4000,
                    "num_tokens": 2,
                    "optim_str_init": "x x",
                    "rand_init": False,
                    "w_circuit_breakers": 1.0,
                    "w_honeypots": None,
                    "log_every": 5,
                },
            }
        ],
    },
}

BENIGN_EVALS = {
    "basic_eval": {
        "tasks": "gsm8k",
    }
}

# ============================================================
# GENERATION
# ============================================================

def main():
    defenses = {}
    pipelines = {}

    # -------- Base (unprotected) models --------
    for model_name in MODELS:
        did = f"base_{model_name}"
        defenses[did] = {
            "script": None,
            "base_model": model_name,
            "output_subdir": did,
            "train_args": {},
        }
        pipelines[did] = [
            {"stage": "attack", "defense": did, "attacks": list(ATTACKS.keys())},
        ]

    # -------- Protected defenses --------
    for model_name in MODELS:
        # for now: if not llama, skip

        for def_name, def_cfg in DEFENSES.items():
            # for now: if not triplet, skip
            if def_name != "cb":
                continue

            for dprof in def_cfg["profiles"]:
                for dpo in DPO_PROFILES:
                    tag = f"{def_name}_{model_name}_{dprof['name']}_{dpo['name']}"

                    train_args = {}
                    train_args.update(dprof["train_args"])
                    train_args.update(dpo["train_args"])

                    defenses[tag] = {
                        "script": def_cfg["script"],
                        "base_model": model_name,
                        "output_subdir": tag,
                        "data": def_cfg["data"],
                        "train_args": train_args,
                    }

                    pipelines[tag] = [
                        {"stage": "train", "defense": tag},
                        {
                            "stage": "attack",
                            "defense": tag,
                            "attacks": list(ATTACKS.keys()),
                        }
                        # ,
                        # {
                            # "stage": "benign_eval",
                            # "defense": tag,
                            # "benign_eval": "basic_eval",
                        # },
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
            "time_attack": "02:00:00",
            "time_benign": "02:00:00",
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

if __name__ == "__main__":
    main()
