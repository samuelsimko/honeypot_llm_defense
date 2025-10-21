# train_lora_refusal_trainer.py

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model
import json
from transformers import PreTrainedTokenizer
from datasets import load_dataset
from typing import List, Dict

# -----------------------------
# 1. Hyperparameters
# -----------------------------
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
max_length = 256
batch_size = 4
num_epochs = 1
learning_rate = 2e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
REFUSAL_RESPONSE = "I'm sorry, I cannot do that."

# -----------------------------
# 2. Load datasets
# -----------------------------
num_samples = 5000  # number of examples to use

# Unsafe: CB
with open("data/circuit_breakers_train.json", "r") as f:
    cb_data = json.load(f)

unsafe_prompts = [item['prompt'] for item in cb_data][:num_samples]
unsafe_responses = [item['output'] for item in cb_data][:num_samples]

ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=42).select(range(num_samples))
safe_prompts = [item['messages'][0]['content'] for item in ultrachat]
safe_responses = [item['messages'][1]['content'] if len(item['messages']) > 1 else "" for item in ultrachat]

print(f"Loaded {len(safe_prompts)} safe examples and {len(unsafe_prompts)} unsafe examples.")

# -----------------------------
# 3. Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_chat_generic(prompts: List[str], responses: List[str], tokenizer: PreTrainedTokenizer, max_length: int = 256):
    if len(prompts) != len(responses):
        raise ValueError("The number of prompts must match the number of responses.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_examples = []
    for prompt, response in zip(prompts, responses):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        full_conversation_tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        prompt_messages = [{"role": "user", "content": prompt}]
        prompt_tokens = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )

        labels = full_conversation_tokens.clone()
        prompt_length = prompt_tokens.shape[-1]
        labels[0, :prompt_length] = -100

        current_len = full_conversation_tokens.shape[-1]
        padding_len = max_length - current_len

        input_ids_padded = torch.nn.functional.pad(
            full_conversation_tokens, (0, padding_len),
            value=tokenizer.pad_token_id
        ).squeeze(0)
        labels_padded = torch.nn.functional.pad(
            labels, (0, padding_len), value=-100
        ).squeeze(0)
        attention_mask = torch.ones_like(full_conversation_tokens)
        attention_mask_padded = torch.nn.functional.pad(
            attention_mask, (0, padding_len), value=0
        ).squeeze(0)

        tokenized_examples.append({
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded
        })
    return tokenized_examples

safe_data = tokenize_chat_generic(safe_prompts, safe_responses, tokenizer=tokenizer)
unsafe_data = tokenize_chat_generic(unsafe_prompts, unsafe_responses, tokenizer=tokenizer)

print("✅ Safe and unsafe data tokenized with assistant-only labels")

class ChatDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

safe_dataset = ChatDataset(safe_data)
unsafe_dataset = ChatDataset(unsafe_data)

# -----------------------------
# 4. Load model + LoRA
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.to(device)
model.train()

# -----------------------------
# 5. Custom Trainer
# -----------------------------
class DualBatchLoader:
    def __init__(self, safe_loader, unsafe_loader):
        self.safe_loader = safe_loader
        self.unsafe_loader = unsafe_loader
    def __iter__(self):
        return zip(self.safe_loader, self.unsafe_loader)
    def __len__(self):
        return min(len(self.safe_loader), len(self.unsafe_loader))

class DualBatchTrainer(Trainer):
    def __init__(self, safe_dataset, unsafe_dataset, layers=[10,20,25,30], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.safe_dataset = safe_dataset
        self.unsafe_dataset = unsafe_dataset
        self.safe_loader = None
        self.unsafe_loader = None
        self.layers = layers
        self.current_step = 0
        self.total_steps = kwargs.get("max_steps", 1000)  # for scheduler

    def get_train_dataloader(self):
        self.safe_loader = DataLoader(
            self.safe_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True
        )
        self.unsafe_loader = DataLoader(
            self.unsafe_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True
        )
        return DualBatchLoader(self.safe_loader, self.unsafe_loader)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        batch_safe, batch_unsafe = inputs
        batch_safe = {k: v.to(model.device) for k, v in batch_safe.items()}
        batch_unsafe = {k: v.to(model.device) for k, v in batch_unsafe.items()}

        # Linear scheduler for safe/unsafe weighting
        alpha = 1 - min(0.5, self.current_step / self.total_steps * 2)
        print(alpha)
        self.current_step += 1

        # Forward pass for LoRA model
        outputs_safe = model(**batch_safe, output_hidden_states=True)
        outputs_unsafe = model(**batch_unsafe, output_hidden_states=True)

        # Forward pass for base model (disable adapters)
        with model.disable_adapter():
            with torch.no_grad():
                base_safe = model(**batch_safe, output_hidden_states=True)
                base_unsafe = model(**batch_unsafe, output_hidden_states=True)

        # L2 loss for safe inputs (mean over tokens and layers)
        l2_loss_safe = torch.mean(torch.stack([
            F.mse_loss(outputs_safe.hidden_states[layer], base_safe.hidden_states[layer], reduction='mean')
            for layer in self.layers
        ]))

        # Cosine similarity loss (minimize ReLU per token, then mean over tokens and layers)
        def relu_cos_loss(h1, h2):
            # h1, h2 shape: (batch, seq_len, hidden)
            cos_sim = F.cosine_similarity(h1, h2, dim=-1)  # (batch, seq_len)
            return torch.mean(F.relu(cos_sim))  # mean over batch and tokens

        cos_loss_unsafe = torch.mean(torch.stack([
            relu_cos_loss(outputs_unsafe.hidden_states[layer], base_unsafe.hidden_states[layer])
            for layer in self.layers
        ]))

        # Weighted total loss
        total_loss = (1 - alpha) * l2_loss_safe + alpha * cos_loss_unsafe
        print(l2_loss_safe, cos_loss_unsafe)

        if return_outputs:
            return total_loss, (outputs_safe, outputs_unsafe)
        return total_loss




# -----------------------------
# 6. Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./cb_lora",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=2,
    learning_rate=learning_rate,
    num_train_epochs=num_epochs,
    bf16=True,
    logging_steps=10,
    save_steps=50,
    max_grad_norm=1.0,
    save_total_limit=2,
    report_to="none"
)

# -----------------------------
# 7. Initialize Trainer
# -----------------------------
trainer = DualBatchTrainer(
    model=model,
    safe_dataset=safe_dataset,
    unsafe_dataset=unsafe_dataset,
    args=training_args,
    tokenizer=tokenizer
)

# -----------------------------
# 8. Train
# -----------------------------
trainer.train()

# -----------------------------
# 9. Save LoRA-adapted model
# -----------------------------
model.save_pretrained("./cb_lora")
