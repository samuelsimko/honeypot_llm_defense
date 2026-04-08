#!/usr/bin/env python3
"""
analyze_actionability_conditioned_on_harm.py

Compute conditional actionability statistics from judged attack outputs.

Main quantity:
  P(GPT-OSS = 1 | HarmBench = 1)

Also reports:
- by attack family
- by model + setting + attack family
- averaged across models for each setting + attack family

Usage:
  python scripts/analyze_actionability_conditioned_on_harm.py \
    --experiment_dir runs/experiments/dpo_base_honeypot_ablation_softprompt
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_model(defense: str) -> str:
    if "llama3_8b" in defense:
        return "llama3_8b"
    if "qwen3_8b" in defense:
        return "qwen3_8b"
    return "unknown"


def parse_setting(defense: str) -> Optional[int]:
    for k in range(10):
        token = f"ablation{k}"
        if token in defense:
            return k
    return None


def parse_attack_group(attack: str) -> str:
    if attack.startswith("soft_prompt"):
        return "soft_prompt"
    if attack.startswith("grpo"):
        return "grpo"
    return "other"


def collect_rows(exp_dir: Path, ensemble_threshold: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for defense_dir in exp_dir.iterdir():
        if not defense_dir.is_dir() or defense_dir.name == "summary":
            continue

        defense_name = defense_dir.name
        attacks_root = defense_dir / "attacks"
        if not attacks_root.exists():
            continue

        for pipeline_dir in attacks_root.iterdir():
            if not pipeline_dir.is_dir():
                continue

            pipeline_name = pipeline_dir.name

            for attack_dir in pipeline_dir.iterdir():
                if not attack_dir.is_dir() or attack_dir.name == "benign_eval":
                    continue

                judged_path = attack_dir / "judged.pkl"
                if not judged_path.exists():
                    continue

                judged = load_pickle(judged_path)

                for rec in judged:
                    j = rec.get("judge_norm", {}) or {}
                    rows.append(
                        {
                            "defense": defense_name,
                            "pipeline": pipeline_name,
                            "attack": attack_dir.name,
                            "attack_group": parse_attack_group(attack_dir.name),
                            "model": parse_model(defense_name),
                            "setting": parse_setting(defense_name),
                            "behavior_id": rec.get("behavior_id"),
                            "variant_id": rec.get("variant_id"),
                            "harmbench": float(j.get("harmbench")) if j.get("harmbench") is not None else np.nan,
                            "wildguard": float(j.get("wildguard")) if j.get("wildguard") is not None else np.nan,
                            "jailjudge_norm": float(j.get("jailjudge_norm")) if j.get("jailjudge_norm") is not None else np.nan,
                            "gpt_oss": float(j.get("gpt_oss")) if j.get("gpt_oss") is not None else np.nan,
                            "ensemble_mean": float(j.get("ensemble_mean")) if j.get("ensemble_mean") is not None else (
                                float(rec.get("ensemble_mean")) if rec.get("ensemble_mean") is not None else np.nan
                            ),
                            "is_harmbench_pos": 1 if j.get("harmbench") == 1 else 0,
                            "is_wildguard_pos": 1 if j.get("wildguard") == 1 else 0,
                            "is_jailjudge_pos": 1 if j.get("jailjudge_norm") == 1 else 0,
                            "is_gpt_oss_pos": 1 if j.get("gpt_oss") == 1 else 0,
                            "is_ensemble_pos": 1 if (
                                j.get("ensemble_mean") is not None and j.get("ensemble_mean") >= ensemble_threshold
                            ) else 0,
                        }
                    )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["setting"] = df["setting"].astype("Int64")
    return df


def conditional_prob(
    df: pd.DataFrame,
    group_cols: List[str],
    given_col: str = "is_harmbench_pos",
    outcome_col: str = "is_gpt_oss_pos",
) -> pd.DataFrame:
    rows = []

    for keys, g in df.groupby(group_cols):
        g_cond = g[g[given_col] == 1]
        if len(g_cond) == 0:
            continue

        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_condition"] = int(len(g_cond))
        row["p_gpt_oss_given_harmbench"] = float(g_cond[outcome_col].mean())
        row["mean_gpt_oss_in_condition"] = float(g_cond["gpt_oss"].mean())
        row["mean_harmbench_in_condition"] = float(g_cond["harmbench"].mean())
        rows.append(row)

    return pd.DataFrame(rows)


def print_section(title: str, df: pd.DataFrame):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if df.empty:
        print("No data.")
    else:
        print(df.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_dir", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--ensemble_threshold", type=float, default=0.8)
    args = ap.parse_args()

    exp_dir = Path(args.experiment_dir)
    out_dir = Path(args.output_dir) if args.output_dir else exp_dir / "summary_conditioned_actionability"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_rows(exp_dir, ensemble_threshold=args.ensemble_threshold)
    if df.empty:
        print("No judged rows found.")
        return

    df.to_csv(out_dir / "all_rows.csv", index=False)

    # ------------------------------------------------------------
    # Tables we actually care about
    # ------------------------------------------------------------
    global_tbl = conditional_prob(
        df,
        group_cols=["attack_group"],
    ).sort_values(["attack_group"])

    by_model_setting_tbl = conditional_prob(
        df[df["setting"].notna()],
        group_cols=["attack_group", "model", "setting"],
    ).sort_values(["attack_group", "model", "setting"])

    by_setting_avg_tbl = conditional_prob(
        df[df["setting"].notna()],
        group_cols=["attack_group", "setting"],
    ).sort_values(["attack_group", "setting"])

    # Pivot tables for readability
    soft_by_model = by_model_setting_tbl[by_model_setting_tbl["attack_group"] == "soft_prompt"].copy()
    grpo_by_model = by_model_setting_tbl[by_model_setting_tbl["attack_group"] == "grpo"].copy()

    soft_avg = by_setting_avg_tbl[by_setting_avg_tbl["attack_group"] == "soft_prompt"].copy()
    grpo_avg = by_setting_avg_tbl[by_setting_avg_tbl["attack_group"] == "grpo"].copy()

    def to_pivot(d: pd.DataFrame) -> pd.DataFrame:
        if d.empty:
            return d
        p = d.pivot(index="model", columns="setting", values="p_gpt_oss_given_harmbench")
        p = p.sort_index(axis=1)
        return p.reset_index()

    soft_pivot = to_pivot(soft_by_model)
    grpo_pivot = to_pivot(grpo_by_model)

    # ------------------------------------------------------------
    # Print clean summaries
    # ------------------------------------------------------------
    print_section(
        "GLOBAL: P(GPT-OSS=1 | HarmBench=1)",
        global_tbl[["attack_group", "n_condition", "p_gpt_oss_given_harmbench"]],
    )

    print_section(
        "AVERAGED ACROSS MODELS: P(GPT-OSS=1 | HarmBench=1) BY SETTING",
        by_setting_avg_tbl[["attack_group", "setting", "n_condition", "p_gpt_oss_given_harmbench"]],
    )

    print_section(
        "SOFT PROMPTS: P(GPT-OSS=1 | HarmBench=1) BY MODEL AND SETTING",
        soft_pivot,
    )

    print_section(
        "GRPO: P(GPT-OSS=1 | HarmBench=1) BY MODEL AND SETTING",
        grpo_pivot,
    )

    print_section(
        "FULL TABLE: BY MODEL / SETTING / ATTACK GROUP",
        by_model_setting_tbl[["attack_group", "model", "setting", "n_condition", "p_gpt_oss_given_harmbench"]],
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    global_tbl.to_csv(out_dir / "global_conditioned.csv", index=False)
    by_model_setting_tbl.to_csv(out_dir / "by_model_setting_conditioned.csv", index=False)
    by_setting_avg_tbl.to_csv(out_dir / "by_setting_avg_conditioned.csv", index=False)

    soft_pivot.to_csv(out_dir / "soft_prompt_pivot.csv", index=False)
    grpo_pivot.to_csv(out_dir / "grpo_pivot.csv", index=False)

    summary = {
        "experiment_dir": str(exp_dir),
        "ensemble_threshold": args.ensemble_threshold,
        "n_rows": int(len(df)),
        "main_quantity": "P(GPT-OSS=1 | HarmBench=1)",
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
