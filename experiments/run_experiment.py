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
from experiments.backends.mock import MockBackend
from experiments.backends.local_gpu import LocalGPUBackend

import hashlib
import json

def stable_hash(obj) -> str:
    """
    Deterministic hash of nested dict / list / str.
    """
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()

def load_attack_config(atk: dict) -> dict:
    if "attack_config" in atk:
        return atk["attack_config"]
    elif "attack_config_path" in atk:
        with open(atk["attack_config_path"]) as f:
            return json.load(f)
    else:
        raise ValueError("Attack must define attack_config or attack_config_path")

def is_stage_done(out_dir: Path, fingerprint: dict) -> bool:
    fp_path = out_dir / "fingerprint.json"
    ready = out_dir / "READY"

    if not (fp_path.exists() and ready.exists()):
        return False

    try:
        old_fp = json.load(open(fp_path))
    except Exception:
        return False

    return old_fp == fingerprint


def write_stage_done(out_dir: Path, fingerprint: dict):
    with open(out_dir / "fingerprint.json", "w") as f:
        json.dump(fingerprint, f, indent=2)
    (out_dir / "READY").touch()


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
    parser.add_argument("--backend", choices=["local", "slurm", "mock", 'local_gpu'], default="local")
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
    elif args.backend == "slurm":
        backend = SlurmBackend(
            partition=cluster["partition"],
            account=cluster["account"],
            gres=cluster["gres"],
        )
    elif args.backend == "mock":
        backend = MockBackend()
    elif args.backend == "local_gpu":
        backend = LocalGPUBackend()
    else:
        raise ValueError(f"Unknown backend: {args.backend}")


    # ============================================================
    # Execute pipeline
    # ============================================================
    pipelines = cfg.get("pipelines")
    if pipelines is None:
        # backward compatibility
        pipelines = {"default": cfg["pipeline"]}

    run_pipelines = cfg.get("run_pipelines", list(pipelines.keys()))
    # Track job IDs for dependencies
    train_jobs: Dict[str, Optional[str]] = {}

    # ============================================================
    # PASS 1: submit ALL training jobs
    # ============================================================

    for pipeline_name, pipeline in pipelines.items():
        if pipeline_name not in run_pipelines:
            continue

        for step in pipeline:
            if step["stage"] != "train":
                continue

            # Only train now

            defense_name = step["defense"]
            if defense_name in train_jobs:
                continue  # already submitted (shared defense)

            ddef = defenses[defense_name]

            # BASE MODEL → no training
            if ddef.get("script") is None:
                train_jobs[defense_name] = None
                continue

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

            fingerprint = {
                "stage": "train",
                "defense": defense_name,
                "script": ddef["script"],
                "base_model": base_model,
                "train_args": ddef.get("train_args", {}),
                "data": ddef.get("data", {}),
            }

            if is_stage_done(out_dir, fingerprint):
                print(f"⏭️  Skipping training {defense_name} (already completed)")
                train_jobs[defense_name] = None
                continue


            job_id = backend.submit(
                name=f"{pipeline_name}_train_{defense_name}",
                command=cmd,
                time=get_time(cluster, "train", "04:00:00"),
                output_log=f"runs/logs/{pipeline_name}_train_{defense_name}.out",
                error_log=f"runs/logs/{pipeline_name}_train_{defense_name}.err",
            )

            train_jobs[defense_name] = job_id
        
    print("✅ Training jobs submitted.")

    # ============================================================
    # PASS 2: submit ALL attack and benign eval jobs
    # ============================================================

    for pipeline_name, pipeline in pipelines.items():
        if pipeline_name not in run_pipelines:
            continue

        print(f"\n🚀 Running pipeline: {pipeline_name}")

        for step in pipeline:
            stage = step["stage"]

            # skip train
            if stage == "train":
                continue

            # ====================================================
            # ATTACK
            # ====================================================

            if stage == "attack":
                defense_name = step["defense"]
                ddef = defenses[defense_name]

                base_model = models[ddef["base_model"]]["hf_id"]
                if ddef.get("script") is not None:
                    lora_path = out_root / ddef["output_subdir"] / "lora_adapter"
                else:
                    lora_path = None

                deps: List[str] = []
                if train_jobs.get(defense_name):
                    deps.append(train_jobs[defense_name])

                for attack_name in step["attacks"]:
                    atk = attacks[attack_name]

                    atk_out = (
                        out_root
                        / ddef["output_subdir"]
                        / "attacks"
                        / pipeline_name
                        / attack_name
                    )
                    atk_out.mkdir(parents=True, exist_ok=True)

                    # --------------------------------------------------
                    # Skip attack if already judged
                    # --------------------------------------------------
                    judged_pkl = atk_out / "judged.pkl"
                    if judged_pkl.exists():
                        print(f"⏭️  Skipping attack {attack_name} for {defense_name} (already judged)")
                        continue

                    fingerprint = {
                        "pipeline": pipeline_name,
                        "stage": "attack",
                        "defense": defense_name,
                        "attack": attack_name,
                        "attack_name": atk["attack_name"],
                        "attack_config": load_attack_config(atk),
                        "model": base_model,
                        "lora_path": str(lora_path) if lora_path is not None else None,
                    }

                    judge_only = step.get("judge_only", False)
                    # if judge, it's never done: always run
                    if is_stage_done(atk_out, fingerprint) and not judge_only:
                        print(f"⏭️  Skipping attack {attack_name} (already completed)")
                        continue

                    attack_cfg = load_attack_config(atk)
                    if "attack_config" in atk:
                        tmp_cfg_path = atk_out / "attack_config.json"
                        with open(tmp_cfg_path, "w") as f:
                            json.dump(attack_cfg, f, indent=2)
                        cfg_path = tmp_cfg_path
                    else:
                        cfg_path = atk["attack_config_path"]

                    cmd = [
                        "python", "attacks/run_attack.py",
                        "--attack", atk["attack_name"],
                        "--model", base_model,
                        "--harmbench-csv", datasets["harmbench_csv"],
                        "--harmbench-targets", datasets["harmbench_targets"],
                        "--output-dir", str(atk_out),
                        "--attack-config-path", str(cfg_path),
                    ]

                    if judge_only:
                        cmd += ["--judge-only"]

                    if ddef.get("script") is not None:
                        lora_path = out_root / ddef["output_subdir"] / "lora_adapter"
                        cmd += ["--lora", str(lora_path)]

                    if "limit" in atk:
                        cmd += ["--limit", str(atk["limit"])]

                    if "num_variants" in atk:
                        cmd += ["--num-variants", str(atk["num_variants"])]

                    if "device" in atk:
                        cmd += ["--device", atk["device"]]

                    backend.submit(
                        name=f"{pipeline_name}_attack_{attack_name}_{defense_name}",
                        command=cmd,
                        time=get_time(cluster, "attack", "05:00:00"),
                        output_log=f"runs/logs/{pipeline_name}_attack_{attack_name}_{defense_name}.out",
                        error_log=f"runs/logs/{pipeline_name}_attack_{attack_name}_{defense_name}.err",
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
                if ddef.get("script") is not None:
                    lora_path = out_root / ddef["output_subdir"] / "lora_adapter"
                else:
                    lora_path = None

                # out_dir = out_root / ddef["output_subdir"] / "benign_eval" / benign_name
                out_dir = out_root / ddef["output_subdir"] / "attacks" / pipeline_name / "benign_eval" / benign_name
                out_dir.mkdir(parents=True, exist_ok=True)

                deps: List[str] = []
                if train_jobs.get(defense_name):
                    deps.append(train_jobs[defense_name])

                cmd = [
                    "python",
                    "benign_capabilities/run_benign_eval.py",
                    "--model", base_model,
                    "--tasks", bcfg["tasks"],
                    "--output-dir", str(out_dir),
                ]

                if lora_path is not None:
                    cmd += ["--lora", str(lora_path)]

                if "limit" in bcfg:
                    cmd += ["--limit", str(bcfg["limit"])]

                fingerprint = {
                    "pipeline": pipeline_name,
                    "stage": "benign_eval",
                    "defense": defense_name,
                    "tasks": bcfg["tasks"],
                    "limit": bcfg.get("limit"),
                    "model": base_model,
                    "lora_path": str(lora_path) if lora_path is not None else None,
                }

                if is_stage_done(out_dir, fingerprint):
                    print(f"⏭️  Skipping benign eval {benign_name}")
                    continue


                backend.submit(
                    name=f"{pipeline_name}_benign_{benign_name}_{defense_name}",
                    command=cmd,
                    time=get_time(cluster, "benign", "04:00:00"),
                    output_log=f"runs/logs/{pipeline_name}_benign_{benign_name}_{defense_name}.out",
                    error_log=f"runs/logs/{pipeline_name}_benign_{benign_name}_{defense_name}.err",
                    depends_on=deps,
                )

            else:
                raise ValueError(f"Unknown pipeline stage: {stage}")

    print("✅ Experiment submission complete.")

    if args.backend == "local_gpu":
        backend.wait_all()
        print("✅ All jobs finished.")


if __name__ == "__main__":
    main()
