#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DATASETS = {
    "harmful_groundtruth": "data/circuit_breakers_train.json",
    "honeypot_normal": "data/honeypots_qwen_fixed.jsonl",
    "ablation_0": "data/honeypot_ablation_0.jsonl",
    "ablation_1": "data/honeypot_ablation_1.jsonl",
    "ablation_2": "data/honeypot_ablation_2.jsonl",
    "ablation_3": "data/honeypot_ablation_3.jsonl",
}

MODELS = {
    "llama3_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen3_8b": "Qwen/Qwen3-8B",
}


def load_json_or_jsonl(path: str) -> List[dict]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            return data if isinstance(data, list) else [data]
        return [json.loads(line) for line in f if line.strip()]


def extract_response(dataset_name: str, rec: dict) -> str | None:
    if dataset_name == "harmful_groundtruth":
        return rec.get("output")
    if dataset_name == "honeypot_normal":
        return rec.get("response")
    if dataset_name.startswith("ablation_"):
        if "response" in rec:
            return rec.get("response")
        if isinstance(rec.get("selected"), dict):
            return rec["selected"].get("response")
    return None


def load_dataset_texts(max_per_dataset: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []

    for name, path in DATASETS.items():
        records = load_json_or_jsonl(path)
        texts = []
        for rec in records:
            text = extract_response(name, rec)
            if text and isinstance(text, str) and text.strip():
                texts.append(text.strip())

        if len(texts) > max_per_dataset:
            texts = rng.sample(texts, max_per_dataset)

        for i, text in enumerate(texts):
            rows.append(
                {
                    "dataset": name,
                    "text_id": f"{name}_{i}",
                    "text": text,
                    "n_chars": len(text),
                }
            )

    df = pd.DataFrame(rows)
    return df


@torch.no_grad()
def embed_texts(
    model_name: str,
    texts: List[str],
    batch_size: int = 8,
    max_length: int = 512,
    dtype: torch.dtype = torch.bfloat16,
) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    all_embs = []

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding with {model_name}"):
        batch = texts[i : i + batch_size]
        toks = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        toks = {k: v.to(model.device) for k, v in toks.items()}

        out = model(**toks, output_hidden_states=False)
        hidden = out.last_hidden_state  # [B, T, H]
        mask = toks["attention_mask"].unsqueeze(-1)  # [B, T, 1]

        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_embs.append(pooled.detach().float().cpu().numpy())

    embs = np.concatenate(all_embs, axis=0)

    del model
    torch.cuda.empty_cache()

    return embs


def run_tsne(embs: np.ndarray, seed: int) -> np.ndarray:
    pca_dim = min(50, embs.shape[1], max(2, embs.shape[0] - 1))
    reduced = PCA(n_components=pca_dim, random_state=seed).fit_transform(embs)

    tsne = TSNE(
        n_components=2,
        perplexity=min(30, max(5, (len(embs) - 1) // 3)),
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    xy = tsne.fit_transform(reduced)
    return xy


def plot_tsne(df_plot: pd.DataFrame, model_key: str, outdir: Path) -> None:
    plt.figure(figsize=(10, 8))

    order = [
        "harmful_groundtruth",
        "honeypot_normal",
        "ablation_0",
        "ablation_1",
        "ablation_2",
        "ablation_3",
    ]

    for dataset in order:
        sub = df_plot[df_plot["dataset"] == dataset]
        if len(sub) == 0:
            continue
        plt.scatter(
            sub["x"],
            sub["y"],
            s=18,
            alpha=0.75,
            label=f"{dataset} (n={len(sub)})",
        )

    plt.title(f"t-SNE of dataset responses using {model_key} embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(markerscale=1.5, fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / f"tsne_{model_key}.png", dpi=220, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=str, default="tsne_plots")
    ap.add_argument("--max-per-dataset", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset texts...")
    df = load_dataset_texts(max_per_dataset=args.max_per_dataset, seed=args.seed)
    df.to_csv(outdir / "sampled_texts.csv", index=False)

    print("\nSample counts:")
    print(df.groupby("dataset").size().sort_index())

    for model_key, model_name in MODELS.items():
        print(f"\n=== {model_key} ===")
        embs = embed_texts(
            model_name=model_name,
            texts=df["text"].tolist(),
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        xy = run_tsne(embs, seed=args.seed)

        df_plot = df.copy()
        df_plot["x"] = xy[:, 0]
        df_plot["y"] = xy[:, 1]
        df_plot.to_csv(outdir / f"tsne_{model_key}.csv", index=False)

        # some quick centroid stats
        centroids = (
            df_plot.groupby("dataset")[["x", "y"]]
            .mean()
            .reset_index()
            .rename(columns={"x": "centroid_x", "y": "centroid_y"})
        )
        centroids.to_csv(outdir / f"centroids_{model_key}.csv", index=False)
        print(centroids.to_string(index=False))

        plot_tsne(df_plot, model_key, outdir)

    print(f"\nSaved outputs to: {outdir}")


if __name__ == "__main__":
    main()
