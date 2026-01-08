#!/usr/bin/env python3
import os
import json
import argparse
import warnings
import shutil
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================================================================
# Extra metrics
# ================================================================
def mean_token_probability(logps):
    logps = np.array(logps)
    return np.exp(logps).mean()

def prefix_logps(logps, k):
    return logps[: min(k, len(logps))]

# ================================================================
# Cache utilities
# ================================================================
def model_exists_in_cache(cache_root: Path, model_id: str) -> bool:
    """
    Check whether a model exists in a HuggingFace-style cache directory.
    """
    hub_dir = cache_root / "hub"
    if not hub_dir.exists():
        return False
    expected = "models--" + model_id.replace("/", "--")
    return any(p.name.startswith(expected) for p in hub_dir.iterdir())


# ================================================================
# Full teacher-forced logprobs (response only)
# ================================================================
@torch.no_grad()
def compute_token_logprobs(model, tokenizer, prompt: str, response: str):
    """
    Compute per-token log probabilities for the RESPONSE only,
    using teacher forcing.
    """
    messages = [{"role": "user", "content": prompt}]

    if tokenizer.chat_template:
        prompt_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_str = prompt

    prompt_ids = tokenizer(
        prompt_str, return_tensors="pt"
    ).input_ids.to(model.device)

    response_ids = tokenizer(
        response,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(model.device)

    # Concatenate prompt + response
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)

    logits = model(input_ids).logits
    start = prompt_ids.shape[1]

    # Align logits so token i predicts response[i]
    response_logits = logits[:, start - 1 : -1, :]
    log_probs = F.log_softmax(response_logits, dim=-1)

    token_logps = log_probs.gather(
        -1, response_ids.unsqueeze(-1)
    ).squeeze(-1)

    return token_logps.cpu().tolist()


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete model files from local cache after finishing each model",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    cache_root = Path.cwd() / "hf_cache"
    cache_root.mkdir(exist_ok=True)

    print(f"[INFO] Using HF cache at: {cache_root.resolve()}")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    with open(args.data_path, "r") as f:
        data = [json.loads(line) for line in f]

    if args.limit:
        data = data[: args.limit]

    print(f"[INFO] Loaded {len(data)} examples")

    # ------------------------------------------------------------------
    # Per-model loop
    # ------------------------------------------------------------------
    for model_id in args.models:
        print(f"\n=== Evaluating {model_id} ===")
        results = []

        cached_before = model_exists_in_cache(cache_root, model_id)
        if cached_before:
            print("[INFO] Model found in local cache")
        else:
            print("[INFO] Model not found — downloading into local cache")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=str(cache_root),
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map=args.device,
                cache_dir=str(cache_root),
            ).eval()

            for ex in tqdm(data):
                try:
                    token_logps = compute_token_logprobs(
                        model,
                        tokenizer,
                        ex["prompt"],
                        ex["response"],
                    )

                    results.append({
                        "model": model_id,
                        "behavior_id": ex.get("id"),
                        "category": ex.get("category"),
                        "strongreject_score": ex.get("strongreject_score"),
                        "num_tokens": len(token_logps),
                        "token_logprobs": token_logps,
                    })

                except Exception as e:
                    warnings.warn(
                        f"Skipping example {ex.get('id', 'UNKNOWN')}: {e}"
                    )

            if not results:
                raise RuntimeError("No successful examples for this model")

            out_path = output_dir / f"{model_id.replace('/', '_')}_logprobs.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(results, f)

            print(f"[INFO] Saved → {out_path}")

        except Exception as e:
            warnings.warn(f"Skipping model {model_id}: {e}")

        finally:
            # Free GPU memory
            try:
                del model
                torch.cuda.empty_cache()
            except Exception:
                pass

            # Optional cleanup
            if args.cleanup and not cached_before:
                hub_dir = cache_root / "hub"
                model_dir = hub_dir / f"models--{model_id.replace('/', '--')}"
                if model_dir.exists():
                    shutil.rmtree(model_dir, ignore_errors=True)
                    print("[INFO] Removed model from local cache")

    print("\nAll done.")


if __name__ == "__main__":
    main()
