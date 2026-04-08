#!/usr/bin/env python3
"""
judged_batch.py

Offline, multi-GPU judge backfill for completed attack runs.

Usage (recommended):
  python judges/judged_batch.py --root runs/experiments/... --gpus 8

Notes:
- Uses multiprocessing start method "spawn" (required for CUDA).
- One worker process per GPU. Each worker sets CUDA_VISIBLE_DEVICES to a single GPU.
- Uses judge_sequence_gpu() so that all HF loads go to the process-visible GPU ("cuda").
"""

import os
import sys
import json
import pickle
import argparse
import gc
from collections import defaultdict
from typing import List, Dict, Any
import multiprocessing as mp
import queue

import numpy as np

# ------------------------------------------------------------
# Make imports robust when running as: python judges/judged_batch.py
# ------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# IMPORTANT: import from judges/judges.py
# Must exist in judges.py:
#   - judge_sequence_gpu
#   - normalize_judge_result
from judges.judges import judge_sequence_gpu, normalize_judge_result


# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ------------------------------------------------------------
# Stats (mirrors run_attack.py)
# ------------------------------------------------------------

def compute_stats(judged: list) -> dict:
    variant_scores = defaultdict(list)
    behavior_scores = defaultdict(lambda: defaultdict(list))

    for r in judged:
        bid = r["behavior_id"]
        for k, v in r.get("judge_norm", {}).items():
            variant_scores[k].append(v)
            behavior_scores[bid][k].append(v)

    stats = {}
    for k, v in variant_scores.items():
        stats[f"variants_mean_{k}"] = float(np.mean(v)) if v else float("nan")

        per_behavior_max = []
        for b in behavior_scores:
            if k in behavior_scores[b] and behavior_scores[b][k]:
                per_behavior_max.append(max(behavior_scores[b][k]))

        stats[f"behavior_max_mean_{k}"] = (
            float(np.mean(per_behavior_max)) if per_behavior_max else float("nan")
        )

    return stats


def save_stats_for_dir(exp_dir: str, judged: list) -> str:
    attack_name = judged[0].get("attack") if judged else None
    fname = f"{attack_name}_stats.json" if attack_name else "stats.json"
    stats_path = os.path.join(exp_dir, fname)

    stats = compute_stats(judged)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats_path


# ------------------------------------------------------------
# Per-experiment judging
# ------------------------------------------------------------

def judge_experiment(exp_dir: str, behavior_label: str):
    """
    Assumes CUDA_VISIBLE_DEVICES is already set in the worker.
    """
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[GPU_VISIBLE={vis}] Judging: {exp_dir}", flush=True)

    completions_path = os.path.join(exp_dir, "completions.pkl")
    judged_path = os.path.join(exp_dir, "judged.pkl")

    if not os.path.exists(completions_path):
        print(f"[GPU_VISIBLE={vis}] Missing completions.pkl, skipping", flush=True)
        return

    if os.path.exists(judged_path):
        print(f"[GPU_VISIBLE={vis}] judged.pkl already exists, skipping", flush=True)
        return

    completions = load_pickle(completions_path)
    prompts = [c["prompt"] for c in completions]
    responses = [c["generated"] for c in completions]

    # ---------------- JUDGING (GPU-correct) ----------------
    raw_results = judge_sequence_gpu(
        prompts,
        responses,
        behavior=behavior_label,
    )

    judged = []
    for c, j in zip(completions, raw_results):
        c = dict(c)
        c["judge_raw"] = j
        c["judge_norm"] = normalize_judge_result(j)
        judged.append(c)

    save_pickle(judged, judged_path)
    stats_path = save_stats_for_dir(exp_dir, judged)

    print(f"[GPU_VISIBLE={vis}] Finished: {exp_dir}", flush=True)
    print(f"[GPU_VISIBLE={vis}] Wrote {judged_path}", flush=True)
    print(f"[GPU_VISIBLE={vis}] Wrote {stats_path}", flush=True)

    # Cleanup
    del completions, judged, raw_results
    try:
        import torch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


# ------------------------------------------------------------
# Worker loop
# ------------------------------------------------------------

def worker(job_queue: mp.Queue, physical_gpu_id: int, behavior_label: str):
    """
    One worker per physical GPU.
    Sets CUDA_VISIBLE_DEVICES so that inside this process:
      - torch sees only one GPU (index 0)
      - transformers device_map="cuda" stays on the visible GPU
    """
    # Must set BEFORE any CUDA init in this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Import torch *after* masking GPUs (safe with spawn)
    import torch
    torch.cuda.set_device(0)

    # Optional: sanity print
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        name = "unknown"
    print(
        f"[WORKER gpu={physical_gpu_id}] visible=CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} name={name}",
        flush=True,
    )

    # Pull jobs until queue is empty (use timeout; mp.Queue.empty() is unreliable)
    while True:
        try:
            exp_dir = job_queue.get(timeout=2.0)
        except (queue.Empty, Exception):
            return

        try:
            judge_experiment(exp_dir, behavior_label)
        except Exception as e:
            print(f"[WORKER gpu={physical_gpu_id}] ERROR in {exp_dir}: {repr(e)}", flush=True)
            # Try to recover GPU memory and continue
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Offline multi-GPU judging")
    parser.add_argument("--root", required=True, help="Root directory to scan")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use (default: all)")
    parser.add_argument("--behavior-label", default="harmful behavior")
    args = parser.parse_args()

    # CUDA + multiprocessing requires spawn
    mp.set_start_method("spawn", force=True)

    # Discover experiments
    exp_dirs = []
    for root, _, files in os.walk(args.root):
        if "completions.pkl" in files and "judged.pkl" not in files:
            exp_dirs.append(root)

    print(f"Found {len(exp_dirs)} experiments to judge", flush=True)
    if not exp_dirs:
        print("Nothing to do.", flush=True)
        return

    # Determine GPU count
    # Import torch here is fine; no CUDA init unless you touch torch.cuda
    import torch
    all_gpus = torch.cuda.device_count()
    num_gpus = args.gpus or all_gpus
    if num_gpus <= 0:
        raise RuntimeError("No GPUs detected (torch.cuda.device_count() == 0).")
    if num_gpus > all_gpus:
        raise ValueError(f"Requested --gpus {num_gpus} but only {all_gpus} available.")

    # Job queue
    job_queue = mp.Queue()
    for d in exp_dirs:
        job_queue.put(d)

    # Spawn one process per GPU (physical IDs 0..num_gpus-1)
    procs = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker, args=(job_queue, gpu_id, args.behavior_label))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("All judging jobs completed", flush=True)


if __name__ == "__main__":
    main()
