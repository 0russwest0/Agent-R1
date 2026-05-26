#!/usr/bin/env bash
set -euo pipefail

export WEBSHOP_DATASET_MODE=full
export WEBSHOP_DATA_DIR="${WEBSHOP_DATA_DIR:-$(pwd)/webshop_data_full}"
export WEBSHOP_INDEX_DIR="${WEBSHOP_INDEX_DIR:-$(pwd)/data/webshop_full}"
export WEBSHOP_SEARCH_TOP_K="${WEBSHOP_SEARCH_TOP_K:-50}"

exec "$(dirname "$0")/run_env_server.sh" "$@"
