#!/usr/bin/env python3
"""
run_experiment.py

Reads an experiment JSON and executes:
  - defense training
  - attacks
  - benign evals

Uses a backend (local or slurm) to submit jobs.
Calls existing scripts; does NOT reimplement logic.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from experiments.backends.local import LocalBackend
from experiments.backends.slurm import SlurmBackend


# ============================================================
# Helpers
# ============================================================

def build_arg_list(args: Dict[str, object]) -> List[str]:
    """
    Convert {key: value} into CLI args:
      {"lr": 1e-4, "epochs": 2} -> ["--lr", "1e-4", "--epochs", "2"]
    """
    out: List[str] = []
    for k, v in args.items():
        if v is None:
            continue
        out.append(f"--{k}")
        out.append(str(v))
    return out


def get_time(cluster: dict, key: str, default: str) -> str:
    """
    Robust time lookup.
    Supports both:
      cluster["time"]["train"]
    and
      cluster["time_train"]
    """
    if "time" in cluster and key in cluster["time"]:
        return cluster["time"][key]
    return cluster.get(f"time_{key}", default)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Experiment JSON")
    parser.add_argument("--backend", choices=["local", "slurm"], default="local")
    args = parser.parse_args()

    # ---------------- Load config ----------------
    with open(args.config, "r") as f:
        cfg = json.load(f)

    meta = cfg["meta"]
    cluster = cfg.get("cluster", {})

    out_root = Path(meta["output_root"]) / meta["experiment_name"]
    out_root.mkdir(parents=True, exist_ok=True)

    models = cfg["models"]
    defenses = cfg.get("defenses", {})
    attacks = cfg.get("attacks", {})
    benign_evals = cfg.get("benign_evals", {})
    datasets = cfg.get("datasets", {})

    # ---------------- Backend ----------------
    if args.backend == "local":
        backend = LocalBackend()
    else:
        backend = SlurmBackend(
            partition=cluster["partition"],
            account=cluster["account"],
            gres=cluster["gres"],
        )

    # Track job IDs for dependencies
    train_jobs: Dict[str, Optional[str]] = {}

    # ============================================================
    # Execute pipeline
    # ============================================================
    for step in cfg["pipeline"]:
        stage = step["stage"]

        # ====================================================
        # TRAIN
        # ====================================================
        if stage == "train":
            defense_name = step["defense"]
            ddef = defenses[defense_name]

            base_model = models[ddef["base_model"]]["hf_id"]
            out_dir = out_root / ddef["output_subdir"]
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "python",
                ddef["script"],
                "--model", base_model,
                "--output_dir", str(out_dir),
            ]

            # training hyperparameters
            cmd += build_arg_list(ddef.get("train_args", {}))

            # data
            data = ddef.get("data", {})
            if "cb_path" in data:
                cmd += ["--cb_path", data["cb_path"]]
            if "honeypot_path" in data:
                cmd += ["--honeypot_path", data["honeypot_path"]]

            job_id = backend.submit(
                name=f"train_{defense_name}",
                command=cmd,
                time=get_time(cluster, "train", "04:00:00"),
                output_log="logs/train_%j.out",
                error_log="logs/train_%j.err",
            )

            train_jobs[defense_name] = job_id

        # ====================================================
        # ATTACK
        # ====================================================
        elif stage == "attack":
            defense_name = step["defense"]
            ddef = defenses[defense_name]

            base_model = models[ddef["base_model"]]["hf_id"]
            lora_path = out_root / ddef["output_subdir"] / "lora_adapter"

            deps: List[str] = []
            if train_jobs.get(defense_name):
                deps.append(train_jobs[defense_name])

            for attack_name in step["attacks"]:
                atk = attacks[attack_name]

                atk_out = out_root / ddef["output_subdir"] / "attacks" / attack_name
                atk_out.mkdir(parents=True, exist_ok=True)

                cmd = [
                    "python", "attacks/run_attack.py",
                    "--attack", atk["attack_name"],
                    "--model", base_model,
                    "--harmbench-csv", datasets["harmbench_csv"],
                    "--harmbench-targets", datasets["harmbench_targets"],
                    "--output-dir", str(atk_out),
                ]

                # optional LoRA
                if lora_path.exists():
                    cmd += ["--lora", str(lora_path)]

                # attack config
                cmd += ["--attack-config-path", atk["attack_config_path"]]

                # limits / debug
                if "limit" in atk:
                    cmd += ["--limit", str(atk["limit"])]

                if "variant_count" in atk:
                    cmd += ["--variant-count", str(atk["variant_count"])]

                if "device" in atk:
                    cmd += ["--device", atk["device"]]

                backend.submit(
                    name=f"attack_{attack_name}_{defense_name}",
                    command=cmd,
                    time=get_time(cluster, "attack", "05:00:00"),
                    output_log="logs/attack_%j.out",
                    error_log="logs/attack_%j.err",
                    depends_on=deps,
                )

        # ====================================================
        # BENIGN EVAL
        # ====================================================
        elif stage == "benign_eval":
            defense_name = step["defense"]
            benign_name = step["benign_eval"]

            ddef = defenses[defense_name]
            bcfg = benign_evals[benign_name]

            base_model = models[ddef["base_model"]]["hf_id"]
            lora_path = out_root / ddef["output_subdir"] / "lora_adapter"

            out_dir = out_root / ddef["output_subdir"] / "benign_eval" / benign_name
            out_dir.mkdir(parents=True, exist_ok=True)

            deps: List[str] = []
            if train_jobs.get(defense_name):
                deps.append(train_jobs[defense_name])

            cmd = [
                "python",
                "benign_capabilities/run_benign_eval.py",
                "--model", base_model,
                "--lora", str(lora_path),
                "--tasks", bcfg["tasks"],
                "--output-dir", str(out_dir),
            ]

            if "limit" in bcfg:
                cmd += ["--limit", str(bcfg["limit"])]

            backend.submit(
                name=f"benign_{benign_name}_{defense_name}",
                command=cmd,
                time=get_time(cluster, "benign", "04:00:00"),
                output_log="logs/benign_%j.out",
                error_log="logs/benign_%j.err",
                depends_on=deps,
            )

        else:
            raise ValueError(f"Unknown pipeline stage: {stage}")

    print("✅ Experiment submission complete.")


if __name__ == "__main__":
    main()