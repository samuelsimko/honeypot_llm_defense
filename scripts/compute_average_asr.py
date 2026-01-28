#!/usr/bin/env python3

import pickle
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import argparse

# ============================================================
# CONFIG
# ============================================================

DEFAULT_ROOT = "runs/attacks_best_defenses"
DEFAULT_METRIC = "ensemble_mean"   # change if needed

# ============================================================
# HELPERS
# ============================================================

def load_judged(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_stats(path: Path):
    with open(path, "r") as f:
        return json.load(f)

# ============================================================
# CORE
# ============================================================

def compute_asr_from_judged(judged, metric):
    """
    ASR = mean_b max_v score(b, v)
    """
    by_behavior = defaultdict(list)

    for row in judged:
        bid = row["behavior_id"]
        score = row["judge_norm"].get(metric)
        if score is not None:
            by_behavior[bid].append(score)

    if not by_behavior:
        return None

    behavior_max = [max(vs) for vs in by_behavior.values()]
    return float(np.mean(behavior_max)), len(by_behavior)

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    root = Path(args.root)
    metric = args.metric

    results = []

    for defense_dir in sorted(root.iterdir()):
        if not defense_dir.is_dir():
            continue

        defense_name = defense_dir.name
        print(f"\n📦 {defense_name}")

        judged_files = list(defense_dir.glob("behavior_*/judged.pkl"))

        if judged_files:
            scores = []
            behaviors = 0

            for jf in judged_files:
                judged = load_judged(jf)
                asr, n_beh = compute_asr_from_judged(judged, metric)
                if asr is not None:
                    scores.append(asr)
                    behaviors += n_beh

            if scores:
                mean_asr = float(np.mean(scores))
                print(f"  ASR ({metric}): {mean_asr:.4f} over ~{behaviors} behaviors")

                results.append({
                    "defense": defense_name,
                    "metric": metric,
                    "asr": mean_asr,
                    "num_behaviors": behaviors,
                })
            else:
                print("  ⚠️ No usable judged data")

        else:
            # Fallback: stats.json at top level
            stats_files = list(defense_dir.glob("*_stats.json"))
            if not stats_files:
                print("  ❌ No judged.pkl or stats.json found")
                continue

            stats = load_stats(stats_files[0])
            key = f"behavior_max_mean_{metric}"
            if key not in stats:
                print(f"  ❌ Metric {key} not in stats")
                continue

            asr = stats[key]
            print(f"  ASR ({metric}): {asr:.4f} (from stats.json)")

            results.append({
                "defense": defense_name,
                "metric": metric,
                "asr": asr,
                "num_behaviors": None,
            })

    # ============================================================
    # SUMMARY
    # ============================================================

    print("\n" + "=" * 60)
    print("📊 AVERAGE ASR ACROSS MODELS")
    print("=" * 60)

    for r in sorted(results, key=lambda x: x["asr"]):
        print(f"{r['defense']:45s}  {r['asr']:.4f}")

    # Optional CSV
    if args.csv:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(args.csv, index=False)
        print(f"\n💾 Saved CSV to {args.csv}")

if __name__ == "__main__":
    main()
