#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export EXP_NAME="${EXP_NAME:-papersearch_gspo}"
export AGENT_R1_GRPO_ROLLOUT_N="${AGENT_R1_GSPO_ROLLOUT_N:-${AGENT_R1_GRPO_ROLLOUT_N:-8}}"

exec bash "$SCRIPT_DIR/run_grpo.sh" \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${AGENT_R1_GSPO_KL_COEF:-0.001}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.use_kl_in_reward=False \
    "$@"
