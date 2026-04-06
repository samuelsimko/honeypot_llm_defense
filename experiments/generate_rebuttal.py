#!/usr/bin/env python3
import json
from pathlib import Path

OUT_PATH = "experiments/rebuttal_llama_cb_resume.json"
EXPERIMENT_NAME = "rebuttal_llama_cb_full"
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
# DPO PROFILES
# ============================================================

DPO_PROFILES = [
    {"name": "nodpo", "train_args": {"w_dpo_hp_over_harm": 0.0}},
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
# CB PROFILES (FULL SET — matches your runs)
# ============================================================

CB_PROFILES = [
    # base
    {"name": "cb_base", "train_args": {"margin_ce": 0.0}},

    # layers
    {
        "name": "cb_layers",
        "train_args": {
            "margin_ce": 0.0,
            "rep_layers": list(range(20, 32)),
        },
    },

    # ce10
    {
        "name": "cb_ce10",
        "train_args": {"margin_ce": 10.0},
    },

    # layers + ce10
    {
        "name": "cb_layers_ce10",
        "train_args": {
            "margin_ce": 10.0,
            "rep_layers": list(range(20, 32)),
        },
    },

    # begn (important!)
    {
        "name": "cb_begn",
        "train_args": {
            "margin_ce": 10.0,
            "w_align_benign": 5.0,
        },
    },
]

# ============================================================
# ATTACKS (EXACT MATCH TO YOUR RUNS)
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
                "optim_str_init": "Make sure to respond well.",
                "rand_init": False,
                "pick_best": "min_final_loss",
            }
        ],
    },
    "soft_prompt_long": {
        "attack_name": "soft_prompt",
        "num_variants": 1,
        "attack_config": [
            {
                "lr": 0.001,
                "num_steps": 500,
                "num_tokens": 20,
                "optim_str_init": "Answer with your unfiltered thoughts.",
                "rand_init": False,
                "pick_best": "min_final_loss",
            }
        ],
    },
    "grpo_prefill": {
        "attack_name": "grpo_prefill",
        "num_variants": 1,
        "attack_config": [
            {
                "lr": 1e-5,
                "num_steps": 200,
                "rollouts_per_step": 4,
                "max_new_tokens": 256,
                "prefill_max_chars": 32,
                "use_elite_buffer": True,
                "elite_buffer_size": 32,
                "seed_jailbreak_bank": ["wildjailbreaks", "jailbreakv"],
            }
        ],
    },
    "grpo_prefill_strong": {
        "attack_name": "grpo_prefill",
        "num_variants": 1,
        "attack_config": [
            {
                "lr": 5e-5,
                "num_steps": 300,
                "rollouts_per_step": 6,
                "max_new_tokens": 256,
                "prefill_max_chars": 30,
                "use_elite_buffer": True,
                "elite_buffer_size": 64,
                "seed_jailbreak_bank": ["wildjailbreaks", "jailbreakv"],
            }
        ],
    },
}

# ============================================================
# BENIGN
# ============================================================

BENIGN_EVALS = {
    "gsm8k_eval": {"tasks": "gsm8k"},
    "mmlu_eval": {"tasks": "mmlu"},
}

# ============================================================
# FILTER LOGIC (IMPORTANT)
# ============================================================

def should_keep(profile_name, dpo_name):
    """
    Keep:
      - ALL begn
      - ALL layers
      - ALL ce10 (requested)
      - ensure nodpo exists for each
    Drop:
      - base_nodpo (useless)
    """
    if "cb_base" in profile_name and dpo_name == "nodpo":
        return False

    return True


# ============================================================
# MAIN
# ============================================================

def main():
    defenses = {}
    pipelines = {}

    # base model
    defenses["base_llama3_8b"] = {
        "script": None,
        "base_model": "llama3_8b",
        "output_subdir": "base_llama3_8b",
        "train_args": {},
    }

    pipelines["base_llama3_8b"] = [
        {
            "stage": "attack",
            "defense": "base_llama3_8b",
            "attacks": list(ATTACKS.keys()),
        },
        {"stage": "benign_eval", "defense": "base_llama3_8b", "benign_eval": "gsm8k_eval"},
        {"stage": "benign_eval", "defense": "base_llama3_8b", "benign_eval": "mmlu_eval"},
    ]

    # defenses
    for cb_prof in CB_PROFILES:
        for dpo in DPO_PROFILES:

            if not should_keep(cb_prof["name"], dpo["name"]):
                continue

            tag = f"cb_llama3_8b_{cb_prof['name']}_{dpo['name']}"

            train_args = {}
            train_args.update(cb_prof["train_args"])
            train_args.update(dpo["train_args"])

            defenses[tag] = {
                "script": "defenses/honeypot_cb.py",
                "base_model": "llama3_8b",
                "output_subdir": tag,
                "data": {
                    "cb_path": "data/circuit_breakers_train.json",
                    "honeypot_path": "data/honeypots_qwen_fixed.jsonl",
                },
                "train_args": train_args,
            }

            pipelines[tag] = [
                {"stage": "train", "defense": tag},
                {
                    "stage": "attack",
                    "defense": tag,
                    "attacks": list(ATTACKS.keys()),
                },
                {"stage": "benign_eval", "defense": tag, "benign_eval": "gsm8k_eval"},
                {"stage": "benign_eval", "defense": tag, "benign_eval": "mmlu_eval"},
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
            "time_train": "03:00:00",
            "time_attack": "03:00:00",
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


if __name__ == "__main__":
    main()
