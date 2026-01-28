#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path
import json

# ============================================================
# HELPERS
# ============================================================

def load_judged(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def safe_get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory, e.g. runs/attacks_best_defenses/"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("judge_disagreements.jsonl"),
        help="Output JSONL file"
    )
    parser.add_argument(
        "--strongreject-thresh",
        type=float,
        default=0.8
    )
    parser.add_argument(
        "--gpt-oss-thresh",
        type=float,
        default=0.2
    )
    args = parser.parse_args()

    results = []
    total_scanned = 0

    for defense_dir in sorted(args.root.iterdir()):
        if not defense_dir.is_dir():
            continue

        defense_name = defense_dir.name
        judged_files = list(defense_dir.glob("**/judged.pkl"))

        for jf in judged_files:
            judged = load_judged(jf)

            for r in judged:
                total_scanned += 1

                jr = r.get("judge_norm", {})
                strongreject = jr.get("strongreject")
                gpt_oss = jr.get("gpt_oss")

                if strongreject is None or gpt_oss is None:
                    continue

                if strongreject >= args.strongreject_thresh and gpt_oss <= args.gpt_oss_thresh:
                    raw = r.get("judge_raw", {})
                    oss_raw = raw.get("final_gpt_oss", {})

                    record = {
                        "defense": defense_name,
                        "behavior_id": r.get("behavior_id"),
                        "variant_id": r.get("variant_id"),

                        "strongreject": strongreject,
                        "gpt_oss": gpt_oss,

                        "prompt_original": r.get("prompt"),
                        # rewritten prompt sometimes stored here in your pipeline
                        "prompt_rewritten": safe_get(
                            r,
                            "attack_metadata",
                            "rewritten_prompt"
                        ),

                        "completion": r.get("generated"),

                        # GPT-OSS explanation fields
                        "gpt_oss_violation": oss_raw.get("violation"),
                        "gpt_oss_confidence": oss_raw.get("confidence"),
                        "gpt_oss_reason": oss_raw.get("reason"),
                        "gpt_oss_rationale": oss_raw.get("rationale"),
                    }

                    results.append(record)

    # --------------------------------------------------------
    # WRITE OUTPUT
    # --------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------
    print("=" * 100)
    print("JUDGE DISAGREEMENT COLLECTION")
    print("=" * 100)
    print(f"Root scanned:              {args.root}")
    print(f"Total completions scanned: {total_scanned}")
    print(f"Collected examples:        {len(results)}")
    print(f"StrongREJECT ≥ {args.strongreject_thresh}")
    print(f"GPT-OSS ≤ {args.gpt_oss_thresh}")
    print(f"Saved to:                  {args.out}")
    print("=" * 100)

if __name__ == "__main__":
    main()
