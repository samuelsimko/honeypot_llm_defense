#!/usr/bin/env python3
"""
summarize_experiment_filtered.py

Like summarize_experiment.py, but:
1) filters out defenses with large benign drops
2) filters out defenses with missing GRPO results
3) prints a compact comparison table where:
   - soft prompts are averaged together
   - each GRPO variant is shown separately

Usage:
  python scripts/summarize_experiment_filtered.py \
    --experiment_dir runs/experiments/rebuttal_llama_cb_full \
    --max_gsm8k_drop 0.05 \
    --max_mmlu_drop 0.05
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


def extract_benign_scores(df_benign: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per defense with gsm8k + mmlu.
    """
    if df_benign.empty:
        return pd.DataFrame(columns=["defense", "gsm8k", "mmlu"])

    out_rows = []

    for defense, g in df_benign.groupby("defense"):
        row = {"defense": defense, "gsm8k": None, "mmlu": None}

        gsm = g[
            (g["task"] == "gsm8k") &
            (g["metric"] == "exact_match_strict_match")
        ]
        if not gsm.empty:
            row["gsm8k"] = float(gsm["value"].iloc[0])

        mmlu = g[
            (g["task"] == "mmlu") &
            (g["metric"] == "acc_none")
        ]
        if not mmlu.empty:
            row["mmlu"] = float(mmlu["value"].iloc[0])

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def parse_model(defense):
    if "llama3_8b" in defense:
        return "llama3_8b"
    if "qwen3_8b" in defense:
        return "qwen3_8b"
    return "unknown"


def parse_regime(defense):
    if defense.startswith("base_"):
        return "base"
    if "nodpo" in defense:
        return "nodpo"
    if "dpo_" in defense:
        return "dpo"
    return "other"


def build_compact_attack_table(df_attack_filtered: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact table:
      - one row for soft_mean = average over soft_prompt_* attacks
      - one row per GRPO attack variant
    """
    if df_attack_filtered.empty:
        return pd.DataFrame()

    metric_cols = [
        c for c in df_attack_filtered.columns
        if c.startswith("behavior_max_mean_")
    ]

    rows = []

    for defense, g in df_attack_filtered.groupby("defense"):
        # average all soft prompt attacks together
        g_soft = g[g["attack"].str.startswith("soft_prompt", na=False)]
        if not g_soft.empty:
            row = {
                "defense": defense,
                "attack_group": "soft_mean",
                "n_attacks": len(g_soft),
            }
            for c in metric_cols:
                row[c] = float(g_soft[c].mean())
            rows.append(row)

        # keep each GRPO attack separate
        g_grpo = g[g["attack"].str.startswith("grpo", na=False)]
        for attack_name, gg in g_grpo.groupby("attack"):
            row = {
                "defense": defense,
                "attack_group": attack_name,
                "n_attacks": len(gg),
            }
            for c in metric_cols:
                row[c] = float(gg[c].mean())
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["model"] = out["defense"].apply(parse_model)
        out["regime"] = out["defense"].apply(parse_regime)
        out = out.sort_values(["model", "attack_group", "behavior_max_mean_ensemble_mean"])
    return out


# -----------------------------
# Main summarizer
# -----------------------------

def summarize_experiment(exp_dir: Path, max_gsm8k_drop: float, max_mmlu_drop: float):
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
            for result_file in defense_dir.rglob("results_*.json"):
                try:
                    data = json.load(open(result_file))
                except Exception:
                    continue

                results = data.get("results", {})
                if not results:
                    continue

                pipeline = "unknown"
                parts = result_file.parts
                if "attacks" in parts:
                    i = parts.index("attacks")
                    if i + 1 < len(parts):
                        pipeline = parts[i + 1]

                eval_name = result_file.parent.name

                for task_name, task_data in results.items():
                    for k, v in task_data.items():
                        if not isinstance(v, (int, float)):
                            continue
                        if "stderr" in k:
                            continue

                        metric = k.replace(",", "_").replace("-", "_")
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

    df_attack = pd.DataFrame(rows_attack)
    df_benign = pd.DataFrame(rows_benign)

    print("\n" + "=" * 80)
    print("RAW ATTACK RESULTS")
    print("=" * 80)
    if not df_attack.empty:
        print(df_attack.sort_values(by=["attack", "defense"]).round(2).to_string(
            index=False,
            columns=[c for c in df_attack.columns if not c.startswith("variant")]
        ))
    else:
        print("No attack data.")

    print("\n" + "=" * 80)
    print("RAW BENIGN RESULTS")
    print("=" * 80)
    print(df_benign.to_string(index=False) if not df_benign.empty else "No benign evals.")

    if df_attack.empty:
        print("No attack data.")
        return

    # -------------------------------------------------
    # Base benign references
    # -------------------------------------------------
    benign_summary = extract_benign_scores(df_benign)

    base_benign = benign_summary[benign_summary["defense"].str.startswith("base_", na=False)].copy()
    base_benign["model"] = base_benign["defense"].apply(parse_model)

    base_ref = (
        base_benign.groupby("model", as_index=False)[["gsm8k", "mmlu"]]
        .mean()
        .rename(columns={"gsm8k": "base_gsm8k", "mmlu": "base_mmlu"})
    )

    benign_summary["model"] = benign_summary["defense"].apply(parse_model)
    benign_summary = benign_summary.merge(base_ref, on="model", how="left")

    benign_summary["gsm8k_drop"] = benign_summary["base_gsm8k"] - benign_summary["gsm8k"]
    benign_summary["mmlu_drop"] = benign_summary["base_mmlu"] - benign_summary["mmlu"]

    # -------------------------------------------------
    # Require GRPO coverage
    # -------------------------------------------------
    grpo_attacks_required = sorted([
        a for a in df_attack["attack"].dropna().unique()
        if a.startswith("grpo")
    ])

    grpo_coverage = (
        df_attack[df_attack["attack"].isin(grpo_attacks_required)]
        .groupby("defense")["attack"]
        .nunique()
        .reset_index(name="n_grpo_attacks")
    )

    benign_summary = benign_summary.merge(grpo_coverage, on="defense", how="left")
    benign_summary["n_grpo_attacks"] = benign_summary["n_grpo_attacks"].fillna(0).astype(int)

    # -------------------------------------------------
    # Filter
    # -------------------------------------------------
    filtered = benign_summary.copy()

    # always keep base defenses
    filtered["keep"] = filtered["defense"].str.startswith("base_", na=False)

    # non-base defenses: need benign drops small enough and all GRPO runs present
    nonbase_mask = ~filtered["defense"].str.startswith("base_", na=False)

    filtered.loc[nonbase_mask, "keep"] = (
        filtered.loc[nonbase_mask, "gsm8k"].notna()
        & filtered.loc[nonbase_mask, "mmlu"].notna()
        & (filtered.loc[nonbase_mask, "gsm8k_drop"] <= max_gsm8k_drop)
        & (filtered.loc[nonbase_mask, "mmlu_drop"] <= max_mmlu_drop)
        & (filtered.loc[nonbase_mask, "n_grpo_attacks"] == len(grpo_attacks_required))
    )

    kept_defenses = set(filtered.loc[filtered["keep"], "defense"].tolist())

    df_attack_filtered = df_attack[df_attack["defense"].isin(kept_defenses)].copy()
    benign_filtered = benign_summary[benign_summary["defense"].isin(kept_defenses)].copy()

    print("\n" + "=" * 80)
    print("FILTERED DEFENSES")
    print("=" * 80)
    print(filtered[[
        "defense", "gsm8k", "mmlu", "gsm8k_drop", "mmlu_drop",
        "n_grpo_attacks", "keep"
    ]].round(4).sort_values(["keep", "defense"], ascending=[False, True]).to_string(index=False))

    print("\n" + "=" * 80)
    print("ATTACK RESULTS AFTER FILTERING")
    print("=" * 80)
    if not df_attack_filtered.empty:
        print(df_attack_filtered.sort_values(by=["attack", "defense"]).round(2).to_string(
            index=False,
            columns=[c for c in df_attack_filtered.columns if not c.startswith("variant")]
        ))
    else:
        print("No attack data after filtering.")

    # -------------------------------------------------
    # Compact table requested
    # -------------------------------------------------
    compact = build_compact_attack_table(df_attack_filtered)

    print("\n" + "=" * 80)
    print("COMPACT TABLE: SOFT PROMPT MEAN + EACH GRPO ATTACK")
    print("=" * 80)
    if compact.empty:
        print("No compact rows.")
    else:
        cols = [
            "defense",
            "model",
            "regime",
            "attack_group",
            "behavior_max_mean_harmbench",
            "behavior_max_mean_strongreject",
            "behavior_max_mean_gpt_oss",
            "behavior_max_mean_ensemble_mean",
        ]
        print(compact[cols].round(3).to_string(index=False))

    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    out_dir = exp_dir / "summary_filtered"
    out_dir.mkdir(exist_ok=True)

    df_attack.to_csv(out_dir / "attacks_raw.csv", index=False)
    df_benign.to_csv(out_dir / "benign_raw.csv", index=False)
    filtered.to_csv(out_dir / "defense_filter_table.csv", index=False)
    df_attack_filtered.to_csv(out_dir / "attacks_filtered.csv", index=False)
    benign_filtered.to_csv(out_dir / "benign_filtered.csv", index=False)
    compact.to_csv(out_dir / "compact_softmean_plus_grpo.csv", index=False)

    with open(out_dir / "filter_config.json", "w") as f:
        json.dump({
            "max_gsm8k_drop": max_gsm8k_drop,
            "max_mmlu_drop": max_mmlu_drop,
            "required_grpo_attacks": grpo_attacks_required,
            "kept_defenses": sorted(list(kept_defenses)),
        }, f, indent=2)

    print(f"\n📁 Saved filtered summary to {out_dir}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiment_dir",
        required=True,
        help="Path to experiment directory",
    )
    ap.add_argument(
        "--max_gsm8k_drop",
        type=float,
        default=0.05,
        help="Maximum allowed absolute GSM8K drop vs base model",
    )
    ap.add_argument(
        "--max_mmlu_drop",
        type=float,
        default=0.05,
        help="Maximum allowed absolute MMLU drop vs base model",
    )
    args = ap.parse_args()

    summarize_experiment(
        Path(args.experiment_dir),
        max_gsm8k_drop=args.max_gsm8k_drop,
        max_mmlu_drop=args.max_mmlu_drop,
    )
