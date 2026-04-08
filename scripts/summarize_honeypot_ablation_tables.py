#!/usr/bin/env python3
"""
summarize_honeypot_ablation_tables.py

Create compact tables for the honeypot-ablation experiment by:
1) merging across models for each setting
2) producing one table for soft-prompt attacks
3) producing one table for GRPO attacks
4) optionally saving LaTeX-friendly CSVs

Usage:
  python scripts/summarize_honeypot_ablation_tables.py \
    --experiment_dir runs/experiments/dpo_base_honeypot_ablation_softprompt

Outputs:
  <experiment_dir>/summary_ablation/attack_rows.csv
  <experiment_dir>/summary_ablation/soft_prompt_by_setting.csv
  <experiment_dir>/summary_ablation/grpo_by_setting.csv
  <experiment_dir>/summary_ablation/soft_prompt_by_model_setting.csv
  <experiment_dir>/summary_ablation/grpo_by_model_setting.csv
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ------------------------------------------------------------
# IO
# ------------------------------------------------------------
def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


# ------------------------------------------------------------
# Parsers
# ------------------------------------------------------------
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
    if defense.startswith("base_"):
        return None
    return None


def parse_attack_group(attack: str) -> str:
    if attack.startswith("soft_prompt"):
        return "soft_prompt"
    if attack.startswith("grpo_prefill_fast"):
        return "grpo"
    return "other"


# ------------------------------------------------------------
# Collect attack rows from experiment dir
# ------------------------------------------------------------
def collect_attack_rows(exp_dir: Path) -> pd.DataFrame:
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
                if not attack_dir.is_dir():
                    continue
                if attack_dir.name == "benign_eval":
                    continue

                stats_path = next(attack_dir.glob("*_stats.json"), None)
                if stats_path is None:
                    continue

                stats = safe_read_json(stats_path)
                if not stats:
                    continue

                row = {
                    "defense": defense_name,
                    "pipeline": pipeline_name,
                    "attack": attack_dir.name,
                    "model": parse_model(defense_name),
                    "setting": parse_setting(defense_name),
                    "attack_group": parse_attack_group(attack_dir.name),
                }
                row.update(stats)
                rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Aggregation helpers
# ------------------------------------------------------------
METRICS = [
    "behavior_max_mean_harmbench",
    "behavior_max_mean_strongreject",
    "behavior_max_mean_wildguard",
    "behavior_max_mean_jailjudge_norm",
    "behavior_max_mean_gpt_oss",
    "behavior_max_mean_ensemble_mean",
]


def aggregate_by_setting(df: pd.DataFrame, attack_group: str) -> pd.DataFrame:
    dfa = df[(df["attack_group"] == attack_group) & (df["setting"].notna())].copy()
    if dfa.empty:
        return pd.DataFrame()

    dfa["setting"] = dfa["setting"].astype(int)

    cols = ["setting"] + [c for c in METRICS if c in dfa.columns]
    out = (
        dfa[cols]
        .groupby("setting", as_index=False)
        .mean(numeric_only=True)
        .sort_values("setting")
    )
    return out


def aggregate_by_model_and_setting(df: pd.DataFrame, attack_group: str) -> pd.DataFrame:
    dfa = df[(df["attack_group"] == attack_group) & (df["setting"].notna())].copy()
    if dfa.empty:
        return pd.DataFrame()

    dfa["setting"] = dfa["setting"].astype(int)

    cols = ["model", "setting"] + [c for c in METRICS if c in dfa.columns]
    out = (
        dfa[cols]
        .groupby(["model", "setting"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["model", "setting"])
    )
    return out


def pretty_percent(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    num_cols = [c for c in out.columns if c not in {"model", "setting"}]
    for c in num_cols:
        out[c] = out[c] * 100.0
    return out


def print_section(title: str, df: pd.DataFrame):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if df.empty:
        print("No data.")
    else:
        print(df.round(2).to_string(index=False))


# ------------------------------------------------------------
# Optional latex helper
# ------------------------------------------------------------
def to_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    keep_cols: List[str],
    rename: Optional[Dict[str, str]] = None,
) -> str:
    x = df[keep_cols].copy()
    if rename:
        x = x.rename(columns=rename)

    # format numbers to 2 decimals
    for c in x.columns:
        if c not in {"model", "setting"}:
            x[c] = x[c].map(lambda v: f"{v:.2f}")

    return x.to_latex(index=False, escape=False, caption=caption, label=label)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_dir", required=True)
    args = ap.parse_args()

    exp_dir = Path(args.experiment_dir)
    out_dir = exp_dir / "summary_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_attack_rows(exp_dir)
    if df.empty:
        print("No attack rows found.")
        return

    df.to_csv(out_dir / "attack_rows.csv", index=False)

    # --------------------------------------------------------
    # Build requested tables
    # --------------------------------------------------------
    soft_setting = pretty_percent(aggregate_by_setting(df, "soft_prompt"))
    grpo_setting = pretty_percent(aggregate_by_setting(df, "grpo"))

    soft_model_setting = pretty_percent(aggregate_by_model_and_setting(df, "soft_prompt"))
    grpo_model_setting = pretty_percent(aggregate_by_model_and_setting(df, "grpo"))

    # Save CSVs
    soft_setting.to_csv(out_dir / "soft_prompt_by_setting.csv", index=False)
    grpo_setting.to_csv(out_dir / "grpo_by_setting.csv", index=False)
    soft_model_setting.to_csv(out_dir / "soft_prompt_by_model_setting.csv", index=False)
    grpo_model_setting.to_csv(out_dir / "grpo_by_model_setting.csv", index=False)

    # Print to terminal
    print_section("SOFT PROMPTS: MERGED ACROSS MODELS BY SETTING", soft_setting)
    print_section("GRPO: MERGED ACROSS MODELS BY SETTING", grpo_setting)
    print_section("SOFT PROMPTS: BY MODEL AND SETTING", soft_model_setting)
    print_section("GRPO: BY MODEL AND SETTING", grpo_model_setting)

    # --------------------------------------------------------
    # Save simple latex tables too
    # --------------------------------------------------------
    keep_cols = [
        "setting",
        "behavior_max_mean_harmbench",
        "behavior_max_mean_strongreject",
        "behavior_max_mean_gpt_oss",
    ]
    rename = {
        "setting": "Setting",
        "behavior_max_mean_harmbench": "HarmBench",
        "behavior_max_mean_strongreject": "StrongREJECT",
        "behavior_max_mean_gpt_oss": "GPT-OSS",
    }

    if not soft_setting.empty:
        latex_soft = to_latex_table(
            soft_setting,
            caption="Mean judge scores for soft-prompt attacks, merged across models, by honeypot setting.",
            label="tab:soft_prompt_ablation_by_setting",
            keep_cols=keep_cols,
            rename=rename,
        )
        with open(out_dir / "soft_prompt_by_setting.tex", "w") as f:
            f.write(latex_soft)

    if not grpo_setting.empty:
        latex_grpo = to_latex_table(
            grpo_setting,
            caption="Mean judge scores for GRPO attacks, merged across models, by honeypot setting.",
            label="tab:grpo_ablation_by_setting",
            keep_cols=keep_cols,
            rename=rename,
        )
        with open(out_dir / "grpo_by_setting.tex", "w") as f:
            f.write(latex_grpo)

    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
