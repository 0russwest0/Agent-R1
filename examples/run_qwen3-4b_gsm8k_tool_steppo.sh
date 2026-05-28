#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/run_qwen3-4b_gsm8k_tool.sh" \
    algorithm.adv_estimator=gae \
    algorithm.gamma="${AGENT_R1_STEPPO_GAMMA:-0.99}" \
    actor_rollout_ref.rollout.n="${AGENT_R1_STEPPO_ROLLOUT_N:-1}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    trainer.experiment_name="${AGENT_R1_EXP_NAME:-qwen3_4b_gsm8k_tool_steppo}" \
    "$@"
