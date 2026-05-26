#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0" .sh)"
LOG_ROOT="${LOG_ROOT:-$(pwd)/logs}"
LOG_DIR="${LOG_DIR:-$LOG_ROOT/papersearch}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${SCRIPT_NAME}_${TIMESTAMP}.log}"

exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export VLLM_USE_V1=1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

SELECTOR_MODEL_PATH=${PAPERSEARCH_SELECTOR_MODEL_PATH:-./checkpoints/selector_Qwen3_8B}
SELECTOR_MODEL_NAME=${PAPERSEARCH_SELECTOR_MODEL_NAME:-selector-qwen-8b}
SELECTOR_HOST=${PAPERSEARCH_SELECTOR_HOST:-0.0.0.0}
SELECTOR_PORT=${PAPERSEARCH_SELECTOR_PORT:-8000}
SELECTOR_DTYPE=${PAPERSEARCH_SELECTOR_DTYPE:-bfloat16}
SELECTOR_TENSOR_PARALLEL_SIZE=${PAPERSEARCH_SELECTOR_TENSOR_PARALLEL_SIZE:-1}
SELECTOR_GPU_MEMORY_UTILIZATION=${PAPERSEARCH_SELECTOR_GPU_MEMORY_UTILIZATION:-0.85}
SELECTOR_MAX_MODEL_LEN=${PAPERSEARCH_SELECTOR_MAX_MODEL_LEN:-32768}

echo "Selector service target: http://localhost:${SELECTOR_PORT}"
echo "Set PAPERSEARCH_SELECTOR_BASE_URL=http://localhost:${SELECTOR_PORT} before training."

vllm serve "$SELECTOR_MODEL_PATH" \
    --runner pooling \
    --host "$SELECTOR_HOST" \
    --port "$SELECTOR_PORT" \
    --served-model-name "$SELECTOR_MODEL_NAME" \
    --dtype "$SELECTOR_DTYPE" \
    --tensor-parallel-size "$SELECTOR_TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$SELECTOR_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$SELECTOR_MAX_MODEL_LEN" \
    --trust-remote-code \
    "$@"
