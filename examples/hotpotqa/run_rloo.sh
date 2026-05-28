#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export EXP_NAME="${EXP_NAME:-hotpotqa_rloo}"
export AGENT_R1_GRPO_ROLLOUT_N="${AGENT_R1_RLOO_ROLLOUT_N:-${AGENT_R1_GRPO_ROLLOUT_N:-8}}"

exec bash "$SCRIPT_DIR/run_grpo.sh" \
    algorithm.adv_estimator=rloo \
    actor_rollout_ref.actor.use_kl_loss=False \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty="${AGENT_R1_RLOO_KL_PENALTY:-kl}" \
    algorithm.kl_ctrl.kl_coef="${AGENT_R1_RLOO_KL_COEF:-0.001}" \
    "$@"
