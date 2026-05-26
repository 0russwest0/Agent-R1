#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/run_qwen3-4b_gsm8k_tool.sh" \
    algorithm.adv_estimator=token_gae \
    actor_rollout_ref.rollout.n="${AGENT_R1_TOKEN_GAE_ROLLOUT_N:-1}" \
    trainer.experiment_name="${AGENT_R1_EXP_NAME:-qwen3_4b_gsm8k_tool_token_gae}" \
    "$@"
