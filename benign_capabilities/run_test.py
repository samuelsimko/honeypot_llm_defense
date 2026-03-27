#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3-32B"
DPO_OFF = "runs/qwen32b_seq_dpo_compare_dpo_off/checkpoint-800"
DPO_ON = "runs/qwen32b_seq_dpo_compare_dpo_on/checkpoint-800"

OUTPUT_ROOT = "benign_capabilities/results_qwen32b_gsm8k"

TASKS = "gsm8k"
LIMIT = None          # set to e.g. 20 for a quick test
DEVICE = "cuda"
DTYPE = "bfloat16"


def run_one(name: str, model: str, lora: str | None, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_args = [
        f"pretrained={model}",
        f"dtype={DTYPE}",
    ]
    if lora is not None:
        model_args.append(f"peft={lora}")

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", ",".join(model_args),
        "--tasks", TASKS,
        "--device", DEVICE,
        "--output_path", output_dir,
    ]

    if LIMIT is not None:
        cmd += ["--limit", str(LIMIT)]

    with open(os.path.join(output_dir, "command.txt"), "w") as f:
        f.write(" ".join(cmd) + "\n")

    print("\n" + "=" * 100)
    print(f"Running: {name}")
    print("Command:")
    print(" ".join(cmd))
    print("=" * 100 + "\n")

    subprocess.run(cmd, check=True)


def main():
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    runs = [
        ("qwen3_32b_base", BASE_MODEL, None),
        ("qwen3_32b_dpo_off", BASE_MODEL, DPO_OFF),
        ("qwen3_32b_dpo_on", BASE_MODEL, DPO_ON),
    ]

    for name, model, lora in runs:
        run_one(
            name=name,
            model=model,
            lora=lora,
            output_dir=os.path.join(OUTPUT_ROOT, name),
        )

    print("\nDone.")
    print(f"Results are in: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
