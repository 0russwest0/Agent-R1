#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0" .sh)"
LOG_ROOT="${LOG_ROOT:-$(pwd)/logs}"
LOG_DIR="${LOG_DIR:-$LOG_ROOT/papersearch_inference}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${SCRIPT_NAME}_${TIMESTAMP}.log}"

exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export PAPER_SEARCH_BASE_URL=${PAPER_SEARCH_BASE_URL:-http://localhost:4000}
export PAPERSEARCH_SELECTOR_BASE_URL=${PAPERSEARCH_SELECTOR_BASE_URL:-http://localhost:8000}

PAPERSEARCH_INFER_DATASET=${PAPERSEARCH_INFER_DATASET:?Set PAPERSEARCH_INFER_DATASET to an external JSONL dataset path.}
PAPERSEARCH_INFER_MODEL_PATH=${PAPERSEARCH_INFER_MODEL_PATH:?Set PAPERSEARCH_INFER_MODEL_PATH to a model or checkpoint path.}
PAPERSEARCH_INFER_OUTPUT_DIR=${PAPERSEARCH_INFER_OUTPUT_DIR:-$(pwd)/results/paper_search/inference}

python3 -m recipes.paper_search.inference.run \
    dataset.path="$PAPERSEARCH_INFER_DATASET" \
    model.path="$PAPERSEARCH_INFER_MODEL_PATH" \
    output.dir="$PAPERSEARCH_INFER_OUTPUT_DIR" \
    "$@"
