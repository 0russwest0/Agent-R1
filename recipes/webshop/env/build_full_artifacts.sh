#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${WEBSHOP_FULL_DATA_DIR:-$(pwd)/webshop_data_full}"
OUTPUT_DIR="${WEBSHOP_FULL_OUTPUT_DIR:-$(pwd)/data/webshop_full}"
THREADS="${WEBSHOP_INDEX_THREADS:-8}"

python -m recipes.webshop.env.full_catalog \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --threads "$THREADS" "$@"

python recipes/webshop/data_preprocess/process_hotpotqa.py \
  --dataset_mode full \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --goals_path "$OUTPUT_DIR/goals.json"
