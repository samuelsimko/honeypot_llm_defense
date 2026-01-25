#!/usr/bin/env python3
"""
Summarize experiment results by scanning an experiment directory.

Works for:
- single-pipeline
- multi-pipeline
- mixed defenses

Directory-driven, not config-driven.
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def safe_read_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def compute_asr(judged_records):
    vals = []
    for r in judged_records:
        j = r.get("judge_norm", {})
        if "is_harmful" in j:
            vals.append(j["is_harmful"])
    return sum(vals) / len(vals) if vals else float("nan")


# -----------------------------
# Main summarizer
# -----------------------------

def summarize_experiment(exp_dir: Path):
    rows_attack = []
    rows_benign = []

    for defense_dir in exp_dir.iterdir():
        if not defense_dir.is_dir():
            continue
        if defense_dir.name == "summary":
            continue

        defense_name = defense_dir.name
        attacks_root = defense_dir / "attacks"
        if not attacks_root.exists():
            continue

        for pipeline_dir in attacks_root.iterdir():
            if not pipeline_dir.is_dir():
                continue

            pipeline_name = pipeline_dir.name

            # -----------------------
            # Attacks
            # -----------------------
            for attack_dir in pipeline_dir.iterdir():
                if not attack_dir.is_dir():
                    continue
                if attack_dir.name == "benign_eval":
                    continue

                judged_path = attack_dir / "judged.pkl"
                if not judged_path.exists():
                    continue

                judged = load_pickle(judged_path)
                asr = compute_asr(judged)

                stats_path = next(attack_dir.glob("*_stats.json"), None)
                stats = safe_read_json(stats_path) if stats_path else {}

                rows_attack.append({
                    "defense": defense_name,
                    "pipeline": pipeline_name,
                    "attack": attack_dir.name,
                    "asr": asr,
                    **stats
                })

            # -----------------------
            # Benign evals
            # -----------------------
            # ======================================================
            # Benign evals (CORRECT FOR NESTED RESULTS FORMAT)
            # ======================================================
            # ======================================================
            # Benign evals (ROBUST: recursive search)
            # ======================================================

            # recursively find all benign result files or stats files
            for result_file in defense_dir.rglob("results_*.json"):
                    try:
                        data = json.load(open(result_file))
                    except Exception:
                        continue

                    results = data.get("results", {})
                    if not results:
                        continue

                    # infer pipeline name if present
                    pipeline = "unknown"
                    parts = result_file.parts
                    if "attacks" in parts:
                        i = parts.index("attacks")
                        if i + 1 < len(parts):
                            pipeline = parts[i + 1]

                    # infer eval name (e.g. gsm8k_full)
                    eval_name = result_file.parent.name

                    for task_name, task_data in results.items():
                        for k, v in task_data.items():
                            if not isinstance(v, (int, float)):
                                continue
                            if "stderr" in k:
                                continue

                            metric = (
                                k.replace(",", "_")
                                .replace("-", "_")
                            )
                            # Dont keep all mmlu_ memtris, just base one
                            if "mmlu_" in task_name and task_name != "mmlu":
                                continue

                            rows_benign.append({
                                "defense": defense_name,
                                "pipeline": pipeline,
                                "eval": eval_name,
                                "task": task_name,
                                "metric": metric,
                                "value": float(v),
                            })

    # -----------------------------
    # Save + print
    # -----------------------------
    df_attack = pd.DataFrame(rows_attack)
    df_benign = pd.DataFrame(rows_benign)
    # order by task, metric, value
    # df_benign = df_benign.sort_values(by=["task", "metric", "value"])

    print("\n" + "=" * 80)
    print("ATTACK SUCCESS RATES")
    print("=" * 80)
    # order per attack, per defense, per asr. Only print columns do NOT have "variant" in the column name
    # order by attack, defense, asr
    df_attack = df_attack.sort_values(by=["attack", "defense", "behavior_max_mean_harmbench"])
    df_attack = df_attack.round(2)
    # only print columns do NOT have "variant" in the column name. Remove "behavior_max_mean_" from the column names. print rounded to two decimal places.
    print(df_attack.to_string(index=False, columns=[c for c in df_attack.columns if not c.startswith("variant")]))
    # print(df_attack.to_string(index=False) if not df_attack.empty else "No attack data.")

    # -----------------------------
    # Aggregate: mean over attacks
    # -----------------------------
    if not df_attack.empty:
        attack_numeric_cols = [
            c for c in df_attack.columns
            if c not in {"defense", "pipeline", "attack"}
            and pd.api.types.is_numeric_dtype(df_attack[c])
        ]

        df_attack_mean = (
            df_attack
            .groupby(["defense", "pipeline"], as_index=False)[attack_numeric_cols]
            .mean()
        )
    else:
        df_attack_mean = pd.DataFrame()
    
    print("\n" + "=" * 80)
    print("MEAN OVER ALL ATTACKS (PER DEFENSE / PIPELINE)")
    print("=" * 80)
    print(df_attack_mean.round(2).to_string(index=False) if not df_attack_mean.empty else "No attack data.")


    print("\n" + "=" * 80)
    print("BENIGN EVAL RESULTS")
    print("=" * 80)
    print(df_benign.to_string(index=False) if not df_benign.empty else "No benign evals.")

    out_dir = exp_dir / "summary"
    out_dir.mkdir(exist_ok=True)

    df_attack.to_csv(out_dir / "attacks.csv", index=False)
    df_benign.to_csv(out_dir / "benign.csv", index=False)

    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "attacks": rows_attack,
            "benign": rows_benign,
        }, f, indent=2)

    print(f"\n📁 Saved summary to {out_dir}")


# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiment_dir",
        required=True,
        help="Path to experiment directory (e.g. runs/experiments/full_pipeline)"
    )
    args = ap.parse_args()

    summarize_experiment(Path(args.experiment_dir))
