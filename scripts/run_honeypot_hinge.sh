#!/bin/bash

python defenses/honeypot_hinge.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --cb_path data/circuit_breakers_train.json \
  --honeypot_path data/cb_train_honeypots.json \
  --ultrachat_samples 1000 \
  --limit_cb 1000 \
  --limit_hp 1000 \
  --batch_size 4 \
  --epochs 1 \
  --lr 1e-4 \
  --w_benign 0.5 \
  --w_refusal 1 \
  --w_honeypot 0.05 \
  --w_harmful 0 \
  --w_align_benign 0.5 \
  --w_align_refusal 0.5 \
  --rep_layers 10 20 30 \
  --margin_ref_gt_hp 4.0 \
  --margin_hp_gt_harm 4.0 \
  --w_margin_ref_hp 0.5 \
  --w_margin_hp_harm 0.5 \
  --output_dir ./honeypot_hinge
