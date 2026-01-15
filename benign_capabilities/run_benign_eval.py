#!/usr/bin/env python3
"""
run_benign_eval.py

Benign capability evaluation using lm-eval-harness.

This runner is intentionally thin:
- No custom scoring logic
- No attacks
- No judging
- Delegates everything to lm_eval

Artifacts:
  output_dir/
    ├── results.json        (lm-eval output)
    ├── meta.json           (run metadata)
    └── command.txt         (exact lm-eval invocation)

Designed to integrate with the full defense/attack pipeline.
"""

import argparse
import json
import os
import subprocess
from datetime import datetime
from typing import Optional


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser("Benign evaluation via lm-eval")

    # Model
    parser.add_argument("--model", required=True,
                        help="HF model name or local base model path")
    parser.add_argument("--lora", default=None,
                        help="Optional LoRA adapter path (defense)")

    # lm-eval
    parser.add_argument("--tasks", default="gsm8k",
                        help="Comma-separated lm-eval tasks")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional example limit per task")

    # Runtime
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")

    # Output
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Construct lm-eval model_args
    # ------------------------------------------------------------
    model_args = [
        f"pretrained={args.model}",
        f"dtype={args.dtype}",
        f"device={args.device}",
    ]

    if args.lora:
        model_args.append(f"peft={args.lora}")

    model_args_str = ",".join(model_args)

    # ------------------------------------------------------------
    # Build lm-eval command
    # ------------------------------------------------------------
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args_str,
        "--tasks", args.tasks,
        "--output_path", args.output_dir,
    ]

    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]

    # Save command for reproducibility
    with open(os.path.join(args.output_dir, "command.txt"), "w") as f:
        f.write(" ".join(cmd) + "\n")

    print("🚀 Running lm-eval:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------
    meta = {
        "model": args.model,
        "lora": args.lora,
        "tasks": args.tasks.split(","),
        "limit": args.limit,
        "device": args.device,
        "dtype": args.dtype,
        "timestamp": datetime.now().isoformat(),
        "runner": "run_benign_eval.py",
    }

    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Benign evaluation complete. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
