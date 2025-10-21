python defenses/honeypot_simple.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --cb_path data/circuit_breakers_train.json \
  --honeypot_path data/cb_train_honeypots.json \
  --ultrachat_samples 5000 \
  --limit_cb 5000 \
  --limit_hp 5000 \
  --batch_size 8 \
  --epochs 1 \
  --lr 2e-4 \
  --w_benign 1.0 \
  --w_honeypot 1.0 \
  --w_harmful -0.5 \
  --output_dir ./honeypot_simple

