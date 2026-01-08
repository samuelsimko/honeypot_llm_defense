"""
Soft Prompt Optimization with Probe-Aware Objective
(White-box adaptive attack against honeypot defenses)
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import random
import torch.nn.functional as F
import tqdm
import logging
from dataclasses import dataclass, field
from typing import List, Union, Dict, Any

import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer


# ============================================================
# Probe utilities
# ============================================================

class LinearProbe(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = torch.nn.Linear(dim, 1)

    def forward(self, h):
        return torch.sigmoid(self.linear(h)).squeeze(-1)


def load_probes(probe_dir, rep_layers, hidden_size, device):
    probes = {}
    for layer in rep_layers:
        path = os.path.join(probe_dir, f"probe_layer{layer}.pt")
        probe = LinearProbe(hidden_size)
        probe.load_state_dict(torch.load(path, map_location=device))
        probe.to(device)
        probe.eval()
        probes[layer] = probe
    return probes


# ============================================================
# Soft prompt config
# ============================================================

@dataclass
class SoftOptConfig:
    num_steps: int = 100
    optim_str_init: str = "x " * 10
    lr: float = 0.01
    device: str = "cuda"
    probe_weight: float = 1.0
    rep_layers: List[int] = field(default_factory=lambda: [10, 20])


# ============================================================
# Probe-aware soft prompt optimization
# ============================================================
def run_soft_opt_with_probing(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    probes: Dict[int, torch.nn.Module],
    messages: Union[str, List[dict]],
    target: str,
    config: SoftOptConfig,
):
    model.eval()
    model.enable_input_require_grads()

    device = config.device
    model = model.to(device)

    # -----------------------
    # Prompt formatting
    # -----------------------
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    messages[-1]["content"] += "{optim_str}"

    template = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    before_str, after_str = template.split("{optim_str}")

    before_ids = tokenizer(before_str, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    after_ids  = tokenizer(after_str,  return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    target_ids = tokenizer(target,     return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    emb = model.get_input_embeddings()
    before_emb = emb(before_ids).detach().clone()
    after_emb  = emb(after_ids).detach().clone()
    target_emb = emb(target_ids).detach().clone()

    # -----------------------
    # Optimized soft prompt
    # -----------------------
    optim_ids = tokenizer(
        config.optim_str_init,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].to(device)

    optim_emb = emb(optim_ids).detach().clone()
    optim_emb.requires_grad_(True)

    optimizer = torch.optim.Adam([optim_emb], lr=config.lr)

    # -----------------------
    # Optimization loop
    # -----------------------
    for step in tqdm.tqdm(range(config.num_steps)):
        optimizer.zero_grad(set_to_none=True)

        input_embeds = torch.cat(
            [before_emb, optim_emb, after_emb, target_emb.detach()],
            dim=1,
        )

        out = model(
            inputs_embeds=input_embeds,
            output_hidden_states=True,
            use_cache=False,
        )

        # -----------------------
        # Target CE loss
        # -----------------------
        logits = out.logits
        shift = input_embeds.shape[1] - target_ids.shape[1]

        shift_logits = logits[
            :, shift - 1 : shift - 1 + target_ids.shape[1], :
        ]

        ce_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            target_ids.view(-1),
        )

        # -----------------------
        # Probe avoidance loss (NO GRADS THROUGH PROBES)
        # -----------------------
        probe_loss = 0.0

        for layer in config.rep_layers:
            h = out.hidden_states[layer]          # [1, S, D]
            h_mean = h.mean(dim=1)                # [1, D]

            with torch.no_grad():
                score = probes[layer](h_mean.float())

            probe_loss += score.mean()

        probe_loss /= len(config.rep_layers)

        total_loss = ce_loss - config.probe_weight * probe_loss
        total_loss.backward()
        optimizer.step()

        # -----------------------
        # Debug
        # -----------------------
        if step == 0 or (step + 1) % 10 == 0 or step == config.num_steps - 1:
            print(
                f"[step {step:03d}] "
                f"CE={ce_loss.item():.4f} | "
                f"Probe={probe_loss.item():.4f} | "
                f"Total={total_loss.item():.4f}"
            )

    return optim_emb.detach().cpu()


# ============================================================
# Benchmark wrapper
# ============================================================

# ============================================================
# Probe training (RESPONSE-BASED, CHAT-AWARE)
# ============================================================

def train_response_probes(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    harmful_data: list,
    honeypot_data: list,
    rep_layers: list,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 2,
    max_prompt_length: int = 512,
    max_gen_tokens: int = 128,
    lr: float = 1e-3,
    num_epochs: int = 1,
    max_steps: int = 20,
):
    os.makedirs(output_dir, exist_ok=True)
    model.eval().to(device)

    hidden_size = model.config.hidden_size
    probes = {l: LinearProbe(hidden_size).to(device) for l in rep_layers}
    opts = {l: torch.optim.Adam(probes[l].parameters(), lr=lr) for l in rep_layers}
    loss_fn = torch.nn.BCELoss()

    def format_chat(prompt):
        msgs = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    dataset = (
        [(x, 1.0) for x in harmful_data] +
        [(x, 0.0) for x in honeypot_data]
    )
    random.shuffle(dataset)

    for epoch in range(num_epochs):
        logging.info(f"🧪 Probe epoch {epoch}")

        for i in tqdm.tqdm(range(0, min(max_steps, len(dataset)), batch_size), desc="Training probes"):
            batch = dataset[i:i + batch_size]
            prompts = [format_chat(x["prompt"]) for x, _ in batch]
            labels = torch.tensor(
                [y for _, y in batch],
                device=device,
                dtype=torch.float32,
            )

            tok = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                gen = model.generate(
                    **tok,
                    max_new_tokens=max_gen_tokens,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                )

            # gen.hidden_states: tuple[layer][B, S, D]
            seq_lens = gen.sequences.shape[1]

            for layer in rep_layers:
                layer_hs = []
                for t in range(len(gen.hidden_states)):
                    layer_hs.append(gen.hidden_states[t][layer])  # [B, 1, D]

                # [B, T, D]
                h = torch.cat(layer_hs, dim=1)
                h_mean = h.mean(dim=1)

                preds = probes[layer](h_mean.float())
                loss = loss_fn(preds, labels)

                opts[layer].zero_grad()
                loss.backward()
                opts[layer].step()

            del gen, tok
            torch.cuda.empty_cache()

    for l, p in probes.items():
        torch.save(p.state_dict(), os.path.join(output_dir, f"probe_layer{l}.pt"))

    return probes


def run_single_benchmark(
    target_model,
    target_tokenizer,
    probes,
    prompt,
    target,
    attack_config_params,
    behavior_id,
    max_generation_length=512,
):
    device = target_model.device
    config = SoftOptConfig(
        device=str(device),
        probe_weight=attack_config_params.get("probe_weight", 1.0),
        rep_layers=attack_config_params.get("rep_layers", [10, 20]),
    )

    optim_emb = run_soft_opt_with_probing(
        model=target_model,
        tokenizer=target_tokenizer,
        probes=probes,
        messages=prompt,
        target=target,
        config=config,
    )

    with torch.no_grad():
        gen = target_model.generate(
            inputs_embeds=optim_emb.to(device=device, dtype=target_model.dtype),
            max_length=max_generation_length,
        )

    text = target_tokenizer.decode(gen[0], skip_special_tokens=True)

    return {
        "behavior_id": behavior_id,
        "prompt": prompt,
        "generated": text,
    }