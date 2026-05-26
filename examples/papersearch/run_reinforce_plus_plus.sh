#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export EXP_NAME="${EXP_NAME:-papersearch_reinforce_plus_plus}"
export AGENT_R1_GRPO_ROLLOUT_N="${AGENT_R1_REINFORCE_PLUS_PLUS_ROLLOUT_N:-${AGENT_R1_GRPO_ROLLOUT_N:-8}}"

exec bash "$ROOT_DIR/examples/run_papersearch_grpo.sh" \
    algorithm.adv_estimator=reinforce_plus_plus \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=mse \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty="${AGENT_R1_REINFORCE_PLUS_PLUS_KL_PENALTY:-kl}" \
    algorithm.kl_ctrl.kl_coef="${AGENT_R1_REINFORCE_PLUS_PLUS_KL_COEF:-0.001}" \
    "$@"
