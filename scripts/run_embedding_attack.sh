source ../venv/bin/activate

# Attack config and data paths
CONFIG_PATH="attacks/embedding/attack_configs.json"
BENCHMARK_CSV="data/harmbench_behaviors_text_val.csv"
TARGETS_JSON="data/harmbench_targets_text.json"
OUTPUT_DIR="results"
# MODEL_PATH="honeypot_hinge_aligned/checkpoint-625/"
MODEL_PATH="samuelsimko/Meta-Llama-3-8B-Instruct-Triplet"


SAFE_MODEL_NAME=$(basename "$MODEL_PATH" | tr '-' '_' | tr '/' '_')
OUTPUT_SUBDIR="attack_results/${SAFE_MODEL_NAME}"

python attacks/embedding/embedding_attack.py \
    --model-path "$MODEL_PATH" \
    --config-path "$CONFIG_PATH" \
    --output-dir "${OUTPUT_DIR}/attack_${SAFE_MODEL_NAME}" \
    --benchmark-data-csv "$BENCHMARK_CSV" \
    --benchmark-targets-json "$TARGETS_JSON"
