#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON in {path}:{line_no}: {e}")
    return rows


def extract_best_candidate(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = rec.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return None

    parsed = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        p = c.get("parsed")
        if isinstance(p, dict):
            parsed.append(c)

    if not parsed:
        return None

    # first parsed candidate; this matches your current generation-only format
    return parsed[0]


def convert_record(rec: Dict[str, Any], level: int) -> Optional[Dict[str, Any]]:
    best = extract_best_candidate(rec)
    if best is None:
        return None

    parsed = best.get("parsed", {}) or {}
    src = rec.get("source_example", {}) or {}

    prompt = rec.get("prompt") or src.get("prompt")
    response = parsed.get("response")
    explanation = parsed.get("explanation")

    if not prompt or not response:
        return None

    all_candidates = []
    for c in rec.get("candidates", []):
        if not isinstance(c, dict):
            continue
        p = c.get("parsed")
        if not isinstance(p, dict):
            continue
        all_candidates.append({
            "attempt": c.get("attempt"),
            "response": p.get("response"),
            "explanation": p.get("explanation"),
            "domain": p.get("domain"),
        })

    return {
        "prompt": prompt,
        "response": response,
        "response_explanation": explanation,
        "domain": parsed.get("domain"),
        "category": rec.get("category") or src.get("category"),
        "level": level,
        "source": "honeypot_ablation",
        "timestamp": rec.get("timestamp"),
        "all_candidates": all_candidates,
    }


def summarize_rows(rows: List[Dict[str, Any]], level: int) -> pd.DataFrame:
    response_lengths = []
    explanation_lengths = []
    candidate_counts = []

    for r in rows:
        response_lengths.append(len(r.get("response") or ""))
        explanation_lengths.append(len(r.get("response_explanation") or ""))
        candidate_counts.append(len(r.get("all_candidates", [])))

    return pd.DataFrame([{
        "level": level,
        "n_examples": len(rows),
        "mean_response_chars": round(float(np.mean(response_lengths)), 2) if response_lengths else np.nan,
        "std_response_chars": round(float(np.std(response_lengths, ddof=1)), 2) if len(response_lengths) > 1 else np.nan,
        "mean_explanation_chars": round(float(np.mean(explanation_lengths)), 2) if explanation_lengths else np.nan,
        "mean_candidates_saved": round(float(np.mean(candidate_counts)), 2) if candidate_counts else np.nan,
    }])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Directory containing level_*_shard_*.jsonl")
    ap.add_argument("--output-dir", required=True, help="Where to write honeypot_ablation_{level}.jsonl")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files_by_level = defaultdict(list)
    for p in sorted(input_dir.glob("level_*_shard_*.jsonl")):
        try:
            parts = p.stem.split("_")
            level = int(parts[1])
            files_by_level[level].append(p)
        except Exception:
            print(f"Skipping unrecognized file: {p.name}")

    if not files_by_level:
        print("No level_*_shard_*.jsonl files found.")
        return

    summary_frames = []

    for level in sorted(files_by_level):
        raw_rows = []
        for p in sorted(files_by_level[level]):
            raw_rows.extend(load_jsonl(p))

        converted_rows = []
        seen_prompts = set()

        for rec in raw_rows:
            out = convert_record(rec, level)
            if out is None:
                continue

            prompt = out["prompt"]
            if prompt in seen_prompts:
                continue
            seen_prompts.add(prompt)
            converted_rows.append(out)

        out_path = output_dir / f"honeypot_ablation_{level}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in converted_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary_df = summarize_rows(converted_rows, level)
        summary_frames.append(summary_df)

        print(f"\n=== LEVEL {level} ===")
        print("Input shards:")
        for p in sorted(files_by_level[level]):
            print(f"  - {p}")
        print(f"Output: {out_path}")
        print(summary_df.to_string(index=False))

    all_summary = pd.concat(summary_frames, ignore_index=True)
    summary_path = output_dir / "honeypot_ablation_summary.csv"
    all_summary.to_csv(summary_path, index=False)

    print("\n=== OVERALL SUMMARY ===")
    print(all_summary.to_string(index=False))
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
