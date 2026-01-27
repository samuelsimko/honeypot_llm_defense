#!/usr/bin/env python3
"""
run_lmeval_parallel.py

Run lm-eval on a set of selected defenses (base / nodpo / dpo) in parallel,
one model per GPU.

Inputs:
- --experiment_dir (expects: <exp>/summary/best_defenses.json)
- or --best_defenses_json explicitly (optional override)

Key features:
- adapter paths resolved relative to experiment_dir
- optional cache warmup to avoid multiple concurrent HF downloads
- clean Ctrl+C handling (terminate children)
"""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_best_defenses(experiment_dir: Path, best_json_override: Optional[Path] = None) -> List[Dict]:
    if best_json_override is not None:
        p = best_json_override
    else:
        p = experiment_dir / "summary" / "best_defenses.json"

    if not p.exists():
        raise FileNotFoundError(f"Could not find best defenses JSON at: {p}")

    with open(p, "r") as f:
        data = json.load(f)

    defenses = data.get("defenses", [])
    if not defenses:
        raise ValueError(f"No defenses found in {p}")

    # If experiment_dir is provided, resolve adapter_path relative to it.
    for d in defenses:
        print(d)
        ap = d.get("adapter_path", None)
        if ap:
            # adapter_path in your json is relative to the experiment directory
            d["adapter_path"] = str((experiment_dir / ap).resolve())

    return defenses


def build_lmeval_cmd(defense: Dict, tasks: str, out_path: Path) -> List[str]:
    # best_defenses.json should include base_model_id; if missing, fall back to model_id or model
    base_model = defense.get("base_model_id") or defense.get("model_id") or defense.get("model")
    if base_model is None:
        raise ValueError(f"Defense missing base model id fields: {defense.keys()}")

    model_args = f"pretrained={base_model}"

    adapter = defense.get("adapter_path")
    if adapter:
        model_args += f",peft={adapter}"

    # IMPORTANT: run lm_eval as module (no accelerate)
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", tasks,
        "--batch_size", "auto",
        "--output_path", str(out_path),
        "--log_samples",
    ]
    return cmd


def warmup_hf_cache(defenses: List[Dict], hf_home: Optional[str] = None):
    """
    Sequentially touch the HF models to pre-download weights/tokenizer.
    This prevents 6-8 simultaneous downloads that can hang/timeout.
    """
    print("\n🔥 Cache warmup (sequential HF downloads)...", flush=True)

    env = os.environ.copy()
    if hf_home:
        env["HF_HOME"] = hf_home

    # Tiny python one-liner that loads tokenizer + model config.
    # (You can switch to AutoModelForCausalLM.from_pretrained if you want full weight download;
    #  this lighter version is often enough to prime artifacts.)
    warm_cmd_tpl = (
        "from transformers import AutoTokenizer, AutoConfig;"
        "m='{m}';"
        "AutoTokenizer.from_pretrained(m, use_fast=True);"
        "AutoConfig.from_pretrained(m);"
        "print('warmed', m)"
    )

    seen = set()
    for d in defenses:
        m = d.get("base_model_id") or d.get("model_id") or d.get("model")
        if not m or m in seen:
            continue
        seen.add(m)

        cmd = [sys.executable, "-c", warm_cmd_tpl.format(m=m)]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)

    print("✅ Cache warmup complete.\n", flush=True)


# ------------------------------------------------------------
# Parallel scheduler
# ------------------------------------------------------------

def run_parallel(defenses: List[Dict], tasks: str, out_dir: Path, gpus: List[int], max_parallel: int,
                 hf_home: Optional[str] = None):
    """
    Simple queue scheduler:
    - launches up to max_parallel jobs at a time
    - each job is pinned to one GPU via CUDA_VISIBLE_DEVICES
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build job list
    jobs = []
    for d in defenses:
        model_name = d.get("model", "unknown")
        regime = d.get("regime", "unknown")
        run_name = f"{model_name}_{regime}"
        out_path = out_dir / f"{run_name}.json"
        jobs.append((run_name, d, out_path))

    # Child proc tracking
    running: List[Dict] = []
    job_idx = 0
    gpu_idx = 0

    def terminate_all():
        for r in running:
            p: subprocess.Popen = r["proc"]
            try:
                p.terminate()
            except Exception:
                pass
        for r in running:
            p: subprocess.Popen = r["proc"]
            try:
                p.kill()
            except Exception:
                pass

    def sigint_handler(sig, frame):
        print("\n🛑 Caught interrupt. Terminating child processes...", flush=True)
        terminate_all()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)

    env_base = os.environ.copy()
    env_base["TOKENIZERS_PARALLELISM"] = "false"
    if hf_home:
        env_base["HF_HOME"] = hf_home

    print(f"\n🚀 Launching lm-eval: {len(jobs)} runs | GPUs={gpus} | max_parallel={max_parallel}", flush=True)

    try:
        while job_idx < len(jobs) or running:
            # Launch new jobs if capacity
            while job_idx < len(jobs) and len(running) < max_parallel:
                run_name, defense, out_path = jobs[job_idx]

                gpu = gpus[gpu_idx % len(gpus)]
                gpu_idx += 1
                job_idx += 1

                env = env_base.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)

                cmd = build_lmeval_cmd(defense, tasks, out_path)

                print(f"\n▶️  GPU {gpu} → {run_name}")
                print(" ".join(cmd), flush=True)

                p = subprocess.Popen(cmd, env=env)
                running.append({"name": run_name, "gpu": gpu, "proc": p})

            # Poll running jobs
            still_running = []
            for r in running:
                p: subprocess.Popen = r["proc"]
                ret = p.poll()
                if ret is None:
                    still_running.append(r)
                else:
                    status = "✅" if ret == 0 else f"❌ (exit={ret})"
                    print(f"{status} Finished {r['name']} on GPU {r['gpu']}", flush=True)
            running = still_running

    except KeyboardInterrupt:
        terminate_all()
        print("🧹 Cleaned up child processes.", flush=True)
        raise

    print("\n✅ All lm-eval runs completed.", flush=True)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_dir", required=True, help="Experiment directory containing summary/best_defenses.json")
    ap.add_argument("--best_defenses_json", default=None, help="Optional override path to best_defenses.json")
    ap.add_argument("--tasks", default="mmlu,gsm8k,arc_challenge,hellaswag,truthfulqa")
    ap.add_argument("--out_dir", default=None, help="Where to write lm-eval outputs (default: <exp>/benign_eval/lmeval)")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="Comma-separated GPU ids to use")
    ap.add_argument("--max_parallel", type=int, default=6, help="How many models to run at once")
    ap.add_argument("--cache_warmup", action="store_true", help="Warm HF cache sequentially before parallel eval")
    ap.add_argument("--hf_home", default=None, help="Optional HF_HOME to share cache across runs/nodes")

    args = ap.parse_args()

    exp_dir = Path(args.experiment_dir).resolve()
    best_override = Path(args.best_defenses_json).resolve() if args.best_defenses_json else None

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (exp_dir / "benign_eval" / "lmeval")

    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if not gpus:
        raise ValueError("No GPUs parsed from --gpus")

    defenses = load_best_defenses(exp_dir, best_override)

    if args.cache_warmup:
        warmup_hf_cache(defenses, hf_home=args.hf_home)

    run_parallel(
        defenses=defenses,
        tasks=args.tasks,
        out_dir=out_dir,
        gpus=gpus,
        max_parallel=min(args.max_parallel, len(gpus)),
        hf_home=args.hf_home,
    )


if __name__ == "__main__":
    main()
