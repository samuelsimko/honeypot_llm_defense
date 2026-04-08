#!/bin/bash
set -euo pipefail

MODEL="huihui-ai/Qwen3-8B-abliterated"
DATA="data/circuit_breakers_train.json"

LEVELS=(0 1 2 3)
TOTAL_PER_LEVEL=4000
NGPU=8

OUT_DIR="ablation_runs"
mkdir -p "$OUT_DIR/logs"

echo "Running 8-GPU sharded ablation..."

# 4 levels × 2 shards = 8 jobs total
SHARDS_PER_LEVEL=2

if (( NGPU != ${#LEVELS[@]} * SHARDS_PER_LEVEL )); then
  echo "Expected NGPU=$(( ${#LEVELS[@]} * SHARDS_PER_LEVEL )) for this setup, got $NGPU"
  exit 1
fi

BASE_PER_SHARD=$((TOTAL_PER_LEVEL / SHARDS_PER_LEVEL))
REMAINDER=$((TOTAL_PER_LEVEL % SHARDS_PER_LEVEL))

job_id=0

for LEVEL in "${LEVELS[@]}"; do
  for (( SHARD=0; SHARD<SHARDS_PER_LEVEL; SHARD++ )); do
    GPU=$job_id

    EXTRA=0
    if (( SHARD < REMAINDER )); then
      EXTRA=1
    fi
    COUNT=$((BASE_PER_SHARD + EXTRA))

    SHARD_LOCAL_START=$((SHARD * BASE_PER_SHARD + (SHARD < REMAINDER ? SHARD : REMAINDER)))
    SHARD_LOCAL_END=$((SHARD_LOCAL_START + COUNT))

    # SAME prompt range for every level
    START=$SHARD_LOCAL_START
    END=$SHARD_LOCAL_END

    OUT_FILE="${OUT_DIR}/level_${LEVEL}_shard_${SHARD}.jsonl"
    LOG_FILE="${OUT_DIR}/logs/l${LEVEL}_s${SHARD}.log"

    rm -f "$OUT_FILE"

    echo "LEVEL $LEVEL | SHARD $SHARD | GPU $GPU | idx [$START,$END)"

    echo CUDA_VISIBLE_DEVICES=$GPU python generation/generate_honeypots_ablation.py \
      --model "$MODEL" \
      --cb_path "$DATA" \
      --output_path "$OUT_FILE" \
      --start_idx "$START" \
      --end_idx "$END" \
      --level "$LEVEL" \
      --temperature 0.7 \
      --max_tries 1

    CUDA_VISIBLE_DEVICES=$GPU python generation/generate_honeypots_ablation.py \
      --model "$MODEL" \
      --cb_path "$DATA" \
      --output_path "$OUT_FILE" \
      --start_idx "$START" \
      --end_idx "$END" \
      --level "$LEVEL" \
      --temperature 0.7 \
      --max_tries 1 \
      > "$LOG_FILE" 2>&1 &

    job_id=$((job_id + 1))
  done
done

wait
echo "All jobs finished."

echo "Concatenating results..."
for LEVEL in "${LEVELS[@]}"; do
  cat "${OUT_DIR}"/level_"${LEVEL}"_shard_*.jsonl > "${OUT_DIR}/out_level_${LEVEL}.jsonl"
done

echo "Done."
