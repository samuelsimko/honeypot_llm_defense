#!/usr/bin/env python3
"""
Print a compact benign-eval summary from lm-eval JSON outputs.

Metrics (explicit):
- ARC-Challenge: acc_norm,none
- HellaSwag: acc,none
- GSM8K: exact_match,strict-match
- MMLU: acc,none (GLOBAL ONLY)
- TruthfulQA:
    - MC1: acc,none
    - MC2: acc,none
    - GEN: bleu_acc,none
"""

import json
from pathlib import Path
import pandas as pd


# -----------------------------
# Config
# -----------------------------

LMEVAL_DIR = Path(
    "runs/experiments/triplet_ablation/benign_eval/lmeval"
)
    # "runs/experiments/dpo_base_threshold_dpo_sweep/benign_eval/lmeval"
     # "runs/experiments/honeypot_structured_ablation/benign_eval/lmeval/"

TASK_SPECS = {
    "arc_challenge": ("ARC-Challenge", ["acc_norm,none"]),
    "hellaswag": ("HellaSwag", ["acc,none"]),
    "gsm8k": ("GSM8K", ["exact_match,strict-match"]),
    "mmlu": ("MMLU", ["acc,none"]),  # global only
    "truthfulqa_mc1": ("TruthfulQA MC1", ["acc,none"]),
    "truthfulqa_mc2": ("TruthfulQA MC2", ["acc,none"]),
    "truthfulqa_gen": ("TruthfulQA GEN", ["bleu_acc,none"]),
}


# -----------------------------
# Helpers
# -----------------------------

def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def extract_metric(results: dict, task: str, keys):
    if task not in results:
        return None
    r = results[task]
    for k in keys:
        if k in r:
            return float(r[k])
    return None


def parse_name(path: Path):
    name = path.stem
    parts = name.split("_")

    model = "_".join(parts[:2])  # llama3_8b / qwen3_8b

    if "dpo" in parts:
        regime = "DPO"
    elif "nodpo" in parts:
        regime = "No-DPO"
    else:
        regime = "Base"

    return model, regime


# -----------------------------
# Main
# -----------------------------

rows = []

for path in sorted(LMEVAL_DIR.glob("*.json")):
    data = load_json(path)
    results = data.get("results", {})

    model, regime = parse_name(path)

    row = {
        "model": model,
        "regime": regime,
        "file": path.name,
    }

    for task, (label, keys) in TASK_SPECS.items():
        row[label] = extract_metric(results, task, keys)

    rows.append(row)


df = pd.DataFrame(rows)

# Convert to percentages
for _, (label, _) in TASK_SPECS.items():
    if label in df:
        df[label] = df[label] * 100.0

print("\n" + "=" * 80)
print("BENIGN PERFORMANCE (LM-EVAL)")
print("=" * 80)
print(df.round(2).to_string(index=False))
