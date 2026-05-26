#!/usr/bin/env bash
set -euo pipefail

HOST="${WEBSHOP_ENV_HOST:-127.0.0.1}"
PORT="${WEBSHOP_ENV_PORT:-4111}"
WORKERS="${WEBSHOP_ENV_WORKERS:-8}"
DATASET_MODE="${WEBSHOP_DATASET_MODE:-full}"

export WEBSHOP_DATASET_MODE="$DATASET_MODE"
export WEBSHOP_ENV_LOG_SEARCH="${WEBSHOP_ENV_LOG_SEARCH:-1}"
if [[ "$DATASET_MODE" == "full" ]]; then
  export WEBSHOP_DATA_DIR="${WEBSHOP_DATA_DIR:-$(pwd)/webshop_data_full}"
  export WEBSHOP_INDEX_DIR="${WEBSHOP_INDEX_DIR:-$(pwd)/data/webshop_full}"
  export WEBSHOP_SEARCH_TOP_K="${WEBSHOP_SEARCH_TOP_K:-50}"
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export JAVA_HOME="${JAVA_HOME:-$CONDA_PREFIX}"
    export PATH="$JAVA_HOME/bin:$PATH"
    export JVM_PATH="${JVM_PATH:-$CONDA_PREFIX/lib/jvm/lib/server/libjvm.so}"
  fi
else
  export WEBSHOP_DATA_DIR="${WEBSHOP_DATA_DIR:-$(pwd)/webshop_data}"
  export WEBSHOP_INDEX_DIR="${WEBSHOP_INDEX_DIR:-$(pwd)/data/webshop/index}"
  export WEBSHOP_SEARCH_TOP_K="${WEBSHOP_SEARCH_TOP_K:-10}"
fi

GUNICORN_ARGS=(
  -w "$WORKERS"
  -k uvicorn.workers.UvicornWorker
  recipe.webshop.env.server:app
  -b "$HOST:$PORT"
  --timeout "${WEBSHOP_ENV_TIMEOUT:-120}"
)

if [[ -n "${WEBSHOP_ENV_ACCESS_LOG:-}" ]]; then
  GUNICORN_ARGS+=(--access-logfile "$WEBSHOP_ENV_ACCESS_LOG")
fi

if [[ -n "${WEBSHOP_ENV_ERROR_LOG:--}" ]]; then
  GUNICORN_ARGS+=(--error-logfile "${WEBSHOP_ENV_ERROR_LOG:--}")
fi

GUNICORN_ARGS+=(--log-level "${WEBSHOP_ENV_LOG_LEVEL:-info}")

echo "Starting WebShop env server with:"
echo "  WEBSHOP_DATASET_MODE=${WEBSHOP_DATASET_MODE}"
echo "  WEBSHOP_DATA_DIR=${WEBSHOP_DATA_DIR}"
echo "  WEBSHOP_INDEX_DIR=${WEBSHOP_INDEX_DIR}"
echo "  WEBSHOP_SEARCH_TOP_K=${WEBSHOP_SEARCH_TOP_K}"

exec gunicorn "${GUNICORN_ARGS[@]}"
