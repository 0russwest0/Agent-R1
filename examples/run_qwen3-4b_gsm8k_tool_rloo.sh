#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/run_qwen3-4b_gsm8k_tool.sh" \
    algorithm.adv_estimator=rloo \
    actor_rollout_ref.rollout.n="${AGENT_R1_RLOO_ROLLOUT_N:-5}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty="${AGENT_R1_RLOO_KL_PENALTY:-kl}" \
    algorithm.kl_ctrl.kl_coef="${AGENT_R1_RLOO_KL_COEF:-0.001}" \
    trainer.experiment_name="${AGENT_R1_EXP_NAME:-qwen3_4b_gsm8k_tool_rloo}" \
    "$@"
