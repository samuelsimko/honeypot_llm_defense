#!/usr/bin/env python3

import subprocess
import os
import json
import queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# ============================================================
# CONFIG — FILL THESE
# ============================================================

BEST_DEFENSES = "best_models.json"
HARM_CSV = "data/harmbench_behaviors_text_val.csv"
HARM_TARGETS = "data/harmbench_targets_text.json"
ATTACK_CONFIG = "grpo.json"

ATTACK = "grpo_prefill"
NUM_VARIANTS = 5

OUT_ROOT = Path("runs/attacks_best_defenses")
LOG_ROOT = Path("logs/attacks_best_defenses")

GPUS = list(range(8))   # 🔥 change if needed
MAX_WORKERS = len(GPUS)

# ============================================================
# GPU QUEUE (dynamic leasing)
# ============================================================

GPU_QUEUE = queue.Queue()
for g in GPUS:
    GPU_QUEUE.put(g)

# ============================================================
# RUN ONE JOB
# ============================================================

def run_one(defense_name, model, lora, idx):
    gpu_id = GPU_QUEUE.get()  # 🔒 blocks until GPU is free

    try:
        out_dir = OUT_ROOT / defense_name / f"behavior_{idx:04d}"
        log_dir = LOG_ROOT / defense_name
        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"behavior_{idx:04d}.log"

        # Optional: skip if already done
        # if (out_dir / "completions.pkl").exists():
        #     return

        cmd = [
            "python", "attacks/run_attack.py",
            "--attack", ATTACK,
            "--model", model,
            "--harmbench-csv", HARM_CSV,
            "--harmbench-targets", HARM_TARGETS,
            "--attack-config-path", ATTACK_CONFIG,
            "--output-dir", str(out_dir),
            "--num-variants", str(NUM_VARIANTS),
            "--behavior-idx", str(idx),
            "--device", "cuda",
        ]

        if lora:
            cmd += ["--lora", lora]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        with open(log_file, "w") as lf:
            subprocess.run(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=env,
            )

    finally:
        GPU_QUEUE.put(gpu_id)  # 🔓 always release GPU

# ============================================================
# MAIN
# ============================================================

def main():
    with open(BEST_DEFENSES) as f:
        defenses = json.load(f)["defenses"]

    df = pd.read_csv(HARM_CSV)
    df = df[df["FunctionalCategory"] == "standard"].reset_index(drop=True)
    num_behaviors = len(df)

    jobs = []
    for d in defenses:
        for idx in range(num_behaviors):
            jobs.append((
                d["defense"],
                d["model"],
                d["adapter_path"],
                idx,
            ))

    print(f"[INFO] Total jobs: {len(jobs)}")
    print(f"[INFO] GPUs available: {GPUS}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(run_one, defense, model, lora, idx)
            for (defense, model, lora, idx) in jobs
        ]

        for f in as_completed(futures):
            f.result()  # propagate crashes immediately

    print("[DONE] All jobs finished successfully")

if __name__ == "__main__":
    main()
