import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# CONFIG
# ==============================
LAYER_ID = 15
MAX_SAMPLES = 40
MAX_LENGTH = 200
BATCH_SIZE = 4

SAVE_TSNE = True
TSNE_OUTPUT_DIR = "analysis/tsne_outputs"

# ==============================
# DATA LOADING
# ==============================
def load_json_or_jsonl(path):
    with open(path, "r") as f:
        text = f.read().strip()

        if text.startswith("["):
            return json.loads(text)  # proper JSON

        # fallback JSONL
        recs = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except:
                pass
        print(len(recs))
        return recs

def load_harmful():
    recs = load_json_or_jsonl("data/circuit_breakers_train.json")
    return [r["prompt"] for r in recs][:MAX_SAMPLES]

def load_honeypot():
    recs = load_json_or_jsonl("data/honeypots_qwen_fixed.jsonl")
    return [r["prompt"] for r in recs][:MAX_SAMPLES]

def load_benign():
    ds = (
        load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        .shuffle(seed=42)
        .select(range(MAX_SAMPLES))
    )

    prompts = []
    for item in ds:
        messages = item["messages"]
        if not messages:
            continue

        user_msg = next(
            (x["content"] for x in messages if x["role"] == "user"),
            None
        )

        if user_msg:
            prompts.append(user_msg)

    return prompts

# ==============================
# TOKENIZATION
# ==============================
def tokenize_batch(tokenizer, prompts):
    return tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

# ==============================
# LOAD MODEL
# ==============================
def load_model(base_model_id, adapter_path=None):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model

# ==============================
# REPRESENTATIONS
# ==============================
@torch.no_grad()
def get_reps(model, tokenizer, prompts):
    reps_layer, reps_final = [], []

    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i+BATCH_SIZE]
        toks = tokenize_batch(tokenizer, batch).to(model.device)

        out = model(
            **toks,
            output_hidden_states=True,
            use_cache=False
        )

        hs = out.hidden_states

        h_layer = hs[LAYER_ID]
        h_final = hs[-1]

        mask = toks["attention_mask"]  # [B, S]

        # flatten correctly
        h_layer = h_layer.reshape(-1, h_layer.shape[-1])
        h_final = h_final.reshape(-1, h_final.shape[-1])

        valid = mask.reshape(-1) > 0

        reps_layer.append(h_layer[valid].cpu())
        reps_final.append(h_final[valid].cpu())

    return torch.cat(reps_layer), torch.cat(reps_final)

# ==============================
# METRICS
# ==============================
def compute_metrics(h_base, h_adapt):
    cos = torch.nn.functional.cosine_similarity(h_base, h_adapt, dim=-1)
    l2 = torch.norm(h_base - h_adapt, dim=-1)

    return {
        "cosine_mean": cos.mean().item(),
        "l2_mean": l2.mean().item()
    }

# ==============================
# LOAD DEFENSE CONFIG
# ==============================
with open("best_models.json", "r") as f:
    CONFIG = json.load(f)["defenses"]

# ==============================
# LOAD DATA
# ==============================
harmful_prompts = load_harmful()
honeypot_prompts = load_honeypot()
benign_prompts = load_benign()

datasets = {
    "harmful": harmful_prompts,
    "honeypot": honeypot_prompts,
    "benign": benign_prompts
}

# ==============================
# MAIN
# ==============================
results = []

for cfg in CONFIG:
    print(f"\n=== {cfg['defense']} ===")

    base_model_id = cfg["model"]
    adapter_path = cfg["adapter_path"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = load_model(base_model_id, None)

    if adapter_path is not None:
        adapted_model = load_model(base_model_id, adapter_path)
    else:
        adapted_model = base_model

    model_result = {
        "defense": cfg["defense"],
        "model": base_model_id,
        "regime": cfg["regime"],
        "metrics": {}
    }

    for name, prompts in datasets.items():
        print(f"  -> {name}")

        h_base_l, h_base_f = get_reps(base_model, tokenizer, prompts)
        h_adapt_l, h_adapt_f = get_reps(adapted_model, tokenizer, prompts)

        metrics_layer = compute_metrics(h_base_l, h_adapt_l)
        metrics_final = compute_metrics(h_base_f, h_adapt_f)

        model_result["metrics"][name] = {
            "layer20": metrics_layer,
            "final": metrics_final
        }

        # ==============================
        # SAVE TSNE
        # ==============================
        if SAVE_TSNE:
            os.makedirs(TSNE_OUTPUT_DIR, exist_ok=True)

            np.save(
                f"{TSNE_OUTPUT_DIR}/{cfg['defense']}_{name}_layer20.npy",
                h_adapt_l.numpy()
            )
            np.save(
                f"{TSNE_OUTPUT_DIR}/{cfg['defense']}_{name}_final.npy",
                h_adapt_f.numpy()
            )

    results.append(model_result)

# ==============================
# SAVE
# ==============================
with open("analysis/rep_drift_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nDone.")
