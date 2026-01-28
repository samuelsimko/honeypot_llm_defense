#!/usr/bin/env python3

import pickle
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import re
import hashlib

# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(s: str) -> str:
    """
    Aggressive-but-safe normalization:
    - lowercase
    - collapse whitespace
    - strip punctuation at ends
    """
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \n\t\r.,;:")
    return s


def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ============================================================
# LOADERS
# ============================================================

def load_any_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# MAIN LOGIC
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="runs/attacks_best_defenses"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=3,
        help="How many example duplicates to print per cluster"
    )
    args = parser.parse_args()

    # hash -> list of (defense, behavior_id, variant_id, path)
    seen = defaultdict(list)

    total = 0

    for defense_dir in sorted(args.root.iterdir()):
        if not defense_dir.is_dir():
            continue

        for pkl_path in defense_dir.glob("**/judged.pkl"):
            try:
                records = load_any_pkl(pkl_path)
            except Exception as e:
                print(f"[WARN] Failed to load {pkl_path}: {e}")
                continue

            for r in records:
                text = r.get("generated")
                if not isinstance(text, str):
                    continue

                norm = normalize_text(text)
                h = hash_text(norm)

                seen[h].append({
                    "defense": defense_dir.name,
                    "behavior_id": r.get("behavior_id"),
                    "variant_id": r.get("variant_id"),
                    "path": str(pkl_path),
                })
                total += 1

    # ============================================================
    # ANALYSIS
    # ============================================================

    duplicate_clusters = {
        h: entries
        for h, entries in seen.items()
        if len(entries) > 1
    }

    # Count cross-defense duplicates only
    cross_defense = {}
    for h, entries in duplicate_clusters.items():
        defenses = {e["defense"] for e in entries}
        if len(defenses) > 1:
            cross_defense[h] = entries

    # ============================================================
    # REPORT
    # ============================================================

    print("\n" + "=" * 100)
    print("DUPLICATE COMPLETION CHECK")
    print("=" * 100)

    print(f"Total completions scanned: {total}")
    print(f"Unique normalized completions: {len(seen)}")

    dup_count = sum(len(v) for v in duplicate_clusters.values())
    print(f"Completions involved in ANY duplicate: {dup_count}")
    print(f"Duplicate clusters (any): {len(duplicate_clusters)}")

    cross_count = sum(len(v) for v in cross_defense.values())
    print(f"Completions duplicated ACROSS DEFENSES: {cross_count}")
    print(f"Cross-defense duplicate clusters: {len(cross_defense)}")

    if total > 0:
        frac = cross_count / total
        print(f"Fraction cross-defense duplicated: {frac:.4%}")

        if frac < 0.01:
            verdict = "✅ very rare — looks healthy"
        elif frac < 0.05:
            verdict = "⚠️ noticeable but probably acceptable"
        else:
            verdict = "🚨 frequent — something systematic is happening"

        print(f"Verdict: {verdict}")

    # ============================================================
    # SHOW EXAMPLES
    # ============================================================

    if cross_defense:
        print("\n" + "-" * 100)
        print("EXAMPLE CROSS-DEFENSE DUPLICATES")
        print("-" * 100)

        shown = 0
        for h, entries in list(cross_defense.items()):
            print(f"\nDuplicate cluster ({len(entries)} occurrences):")
            for e in entries[:args.max_examples]:
                print(
                    f"  - defense={e['defense']}, "
                    f"behavior={e['behavior_id']}, "
                    f"variant={e['variant_id']}"
                )
            shown += 1
            if shown >= args.max_examples:
                break

    print("\nDone.")


if __name__ == "__main__":
    main()
