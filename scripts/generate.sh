BENCHMARK_CSV="/data/samuel_simko/HarmBench/data/behavior_datasets/harmbench_behaviors_text_val.csv"

python generation/generate_categories.py \
  --model-path mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated \
  --benchmark-data-csv $BENCHMARK_CSV \
  --output-path data/various_harmful_dataset.jsonl \
  --device cuda:0 \
  --max-length 256 \
  --limit 50

