python attack.py \
    --model-path samuelsimko/Meta-Llama-3-8B-Instruct-Triplet-Adv \
    --cb-path ../../data/circuit_breakers_train.json \
    --benchmark-data-csv ../../data/harmbench_behaviors_text_all.csv \
    --limit-completions 2 \
    --benchmark-targets-json ../../data/harmbench_targets_text.json

# --model-path meta-llama/Meta-Llama-3-8B-Instruct \
