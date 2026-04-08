#!/usr/bin/env python3
"""
judge_honeypot_ablation_datasets.py

Offline, multi-GPU judging for exported honeypot ablation datasets.

Expected inputs:
  data/honeypot_ablation_0.jsonl
  data/honeypot_ablation_1.jsonl
  data/honeypot_ablation_2.jsonl
  data/honeypot_ablation_3.jsonl

Usage:
  python scripts/judge_honeypot_ablation_datasets.py \
    --input-dir data \
    --output-dir judged_ablation \
    --pattern "honeypot_ablation_*.jsonl" \
    --limit 200 \
    --gpus 4

Notes:
- Uses multiprocessing start method "spawn" (required for CUDA).
- One worker process per GPU.
- IMPORTANT: judge imports are lazy, inside the worker/job function,
  so CUDA_VISIBLE_DEVICES is respected.
"""

import os
import sys
import json
import glob
import argparse
import gc
import multiprocessing as mp
import queue
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional

import numpy as np


# ------------------------------------------------------------
# Make imports robust when running as: python scripts/...
# ------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------
def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)

        if head == "[":
            data = json.load(f)
            return data if isinstance(data, list) else [data]

        for line in f:
            s = line.strip()
            if s:
                records.append(json.loads(s))
    return records


def write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ------------------------------------------------------------
# Field extraction
# ------------------------------------------------------------
def extract_prompt(rec: Dict[str, Any]) -> str:
    return rec.get("prompt", "")


def extract_response(rec: Dict[str, Any]) -> str:
    # Exported ablation format
    if "response" in rec and rec["response"] is not None:
        return rec["response"]

    # Raw generation format fallback
    if "selected" in rec and isinstance(rec["selected"], dict):
        return rec["selected"].get("response", "")

    # Candidate fallback
    if "candidates" in rec and isinstance(rec["candidates"], list) and rec["candidates"]:
        cand0 = rec["candidates"][0]
        if isinstance(cand0, dict):
            if "parsed" in cand0 and isinstance(cand0["parsed"], dict):
                return cand0["parsed"].get("response", "")
            return cand0.get("response", "")

    return ""


def extract_level(rec: Dict[str, Any], fallback_from_name: Optional[int] = None) -> Optional[int]:
    if "level" in rec:
        try:
            return int(rec["level"])
        except Exception:
            return fallback_from_name
    return fallback_from_name


# ------------------------------------------------------------
# Stats
# ------------------------------------------------------------
def compute_summary(judged_rows: List[Dict[str, Any]], input_file: str, limit: Optional[int]) -> Dict[str, Any]:
    judge_keys = set()
    for r in judged_rows:
        judge_keys.update((r.get("judge_norm") or {}).keys())

    summary: Dict[str, Any] = {
        "input_file": input_file,
        "n_rows": len(judged_rows),
        "limit": limit,
    }

    if judged_rows:
        levels = [r.get("level") for r in judged_rows if r.get("level") is not None]
        if levels:
            summary["level"] = int(levels[0])

    for k in sorted(judge_keys):
        vals = [r["judge_norm"][k] for r in judged_rows if r.get("judge_norm") and k in r["judge_norm"]]
        if vals:
            summary[f"mean_{k}"] = float(np.mean(vals))
            summary[f"std_{k}"] = float(np.std(vals))
        else:
            summary[f"mean_{k}"] = None
            summary[f"std_{k}"] = None

    # Some useful aliases
    if "mean_strongreject" in summary:
        summary["mean_actionability_proxy"] = summary["mean_strongreject"]
    if "mean_gpt_oss" in summary:
        summary["mean_gpt_oss_actionability"] = summary["mean_gpt_oss"]

    return summary


# ------------------------------------------------------------
# Per-file judging
# ------------------------------------------------------------
def judge_dataset_file(input_path: str, output_dir: str, limit: Optional[int], behavior_label: str):
    """
    Assumes CUDA_VISIBLE_DEVICES is already set in the worker.
    """
    # IMPORTANT: lazy import after CUDA_VISIBLE_DEVICES is set
    from judges.judges import judge_sequence_gpu, normalize_judge_result

    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[GPU_VISIBLE={vis}] Judging file: {input_path}", flush=True)

    input_name = Path(input_path).stem
    judged_path = os.path.join(output_dir, f"{input_name}.judged.jsonl")
    summary_path = os.path.join(output_dir, f"{input_name}.summary.json")

    if os.path.exists(judged_path):
        print(f"[GPU_VISIBLE={vis}] Output already exists, skipping: {judged_path}", flush=True)
        return

    rows = load_json_or_jsonl(input_path)
    if limit is not None:
        rows = rows[:limit]

    # Try to infer level from filename if missing
    inferred_level = None
    try:
        inferred_level = int(input_name.split("_")[-1])
    except Exception:
        inferred_level = None

    prompts = [extract_prompt(r) for r in rows]
    responses = [extract_response(r) for r in rows]

    raw_results = judge_sequence_gpu(
        prompts,
        responses,
        behavior=behavior_label,
    )

    judged_rows: List[Dict[str, Any]] = []
    for r, j in zip(rows, raw_results):
        rr = dict(r)
        rr["level"] = extract_level(rr, inferred_level)
        rr["judge_raw"] = j
        rr["judge_norm"] = normalize_judge_result(j)
        judged_rows.append(rr)

    write_jsonl(judged_rows, judged_path)

    summary = compute_summary(
        judged_rows=judged_rows,
        input_file=input_path,
        limit=limit,
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[GPU_VISIBLE={vis}] Finished: {input_path}", flush=True)
    print(f"[GPU_VISIBLE={vis}] Wrote {judged_path}", flush=True)
    print(f"[GPU_VISIBLE={vis}] Wrote {summary_path}", flush=True)

    # Cleanup
    del rows, prompts, responses, raw_results, judged_rows
    try:
        import torch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


# ------------------------------------------------------------
# Worker loop
# ------------------------------------------------------------
def worker(job_queue: mp.Queue, physical_gpu_id: int, output_dir: str, limit: Optional[int], behavior_label: str):
    """
    One worker per physical GPU.
    Sets CUDA_VISIBLE_DEVICES so that inside this process:
      - torch sees only one GPU (index 0)
      - HF models stay on the process-visible GPU
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    torch.cuda.set_device(0)

    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        name = "unknown"

    print(
        f"[WORKER gpu={physical_gpu_id}] visible=CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} name={name}",
        flush=True,
    )

    while True:
        try:
            input_path = job_queue.get(timeout=2.0)
        except (queue.Empty, Exception):
            return

        try:
            judge_dataset_file(
                input_path=input_path,
                output_dir=output_dir,
                limit=limit,
                behavior_label=behavior_label,
            )
        except Exception as e:
            print(f"[WORKER gpu={physical_gpu_id}] ERROR in {input_path}: {repr(e)}", flush=True)
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Offline multi-GPU judging for honeypot ablation datasets")
    parser.add_argument("--input-dir", required=True, help="Directory containing honeypot_ablation_*.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory to write judged outputs")
    parser.add_argument("--pattern", default="honeypot_ablation_*.jsonl", help="Glob pattern inside input-dir")
    parser.add_argument("--limit", type=int, default=200, help="Judge only the first N rows per file")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use (default: all visible)")
    parser.add_argument("--behavior-label", default="harmful behavior")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    print(f"Found {len(files)} files", flush=True)
    for f in files:
        print(f" - {f}", flush=True)

    if not files:
        print("Nothing to do.", flush=True)
        return

    import torch
    all_gpus = torch.cuda.device_count()
    num_gpus = args.gpus or all_gpus
    if num_gpus <= 0:
        raise RuntimeError("No GPUs detected (torch.cuda.device_count() == 0).")
    if num_gpus > all_gpus:
        raise ValueError(f"Requested --gpus {num_gpus} but only {all_gpus} available.")

    job_queue = mp.Queue()
    for f in files:
        job_queue.put(f)

    procs = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker,
            args=(job_queue, gpu_id, args.output_dir, args.limit, args.behavior_label),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # Aggregate overview
    overview_rows = []
    for f in files:
        name = Path(f).stem
        summary_path = os.path.join(args.output_dir, f"{name}.summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as fh:
                overview_rows.append(json.load(fh))

    overview_path = os.path.join(args.output_dir, "overview.json")
    with open(overview_path, "w", encoding="utf-8") as f:
        json.dump(overview_rows, f, indent=2)

    print(f"Saved overview to {overview_path}", flush=True)
    print("All judging jobs completed", flush=True)


if __name__ == "__main__":
    main()
