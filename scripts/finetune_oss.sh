python defense_finetune.py \
  --ref-model openai/gpt-oss-20b \
  --data data/circuit_breakers_train.json \
  --save-path ./defended_model_sft_early \
  --epochs 1 \
  --batch-size 4

# --ref-model meta-llama/Meta-Llama-3-8B-Instruct \
