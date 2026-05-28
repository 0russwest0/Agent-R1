#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0" .sh)"
LOG_ROOT="${LOG_ROOT:-$(pwd)/logs}"
LOG_DIR="${LOG_DIR:-$LOG_ROOT/webshop}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${SCRIPT_NAME}_${TIMESTAMP}.log}"

exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
set -x

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VLLM_USE_V1=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export HYDRA_FULL_ERROR=1
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
export WEBSHOP_ENV_BASE_URL="${WEBSHOP_ENV_BASE_URL:-http://127.0.0.1:4111}"

AGENT_R1_GRPO_ROLLOUT_N="${AGENT_R1_GRPO_ROLLOUT_N:-8}"
# Keep the effective unique-task budget near 128: train_batch_size * rollout.n ~= 128.
WEBSHOP_GRPO_BASE_TRAIN_BATCH="${WEBSHOP_GRPO_BASE_TRAIN_BATCH:-128}"
WEBSHOP_GRPO_BASE_LOG_PROB_MICRO_BATCH="${WEBSHOP_GRPO_BASE_LOG_PROB_MICRO_BATCH:-32}"
WEBSHOP_TRAIN_BATCH_SIZE="$((WEBSHOP_GRPO_BASE_TRAIN_BATCH / AGENT_R1_GRPO_ROLLOUT_N))"
WEBSHOP_LOG_PROB_MICRO_BATCH="$((WEBSHOP_GRPO_BASE_LOG_PROB_MICRO_BATCH / AGENT_R1_GRPO_ROLLOUT_N))"
if [[ "$WEBSHOP_TRAIN_BATCH_SIZE" -lt 1 ]]; then
    echo "❌ WEBSHOP_GRPO_BASE_TRAIN_BATCH ($WEBSHOP_GRPO_BASE_TRAIN_BATCH) must be >= AGENT_R1_GRPO_ROLLOUT_N ($AGENT_R1_GRPO_ROLLOUT_N)." >&2
    exit 1
fi
if [[ "$WEBSHOP_LOG_PROB_MICRO_BATCH" -lt 1 ]]; then
    WEBSHOP_LOG_PROB_MICRO_BATCH=1
fi

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/recipes/webshop/base.yaml"

WEBSHOP_MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
WEBSHOP_MAX_PROMPT_LEN="${WEBSHOP_MAX_PROMPT_LEN:-8192}"
WEBSHOP_MAX_RESPONSE_LEN="${WEBSHOP_MAX_RESPONSE_LEN:-4096}"
WEBSHOP_DATA_ROOT="${WEBSHOP_DATA_ROOT:-$PROJECT_DIR/data/webshop_full}"
WEBSHOP_TRAIN_PATH="${WEBSHOP_TRAIN_PATH:-$WEBSHOP_DATA_ROOT/train.parquet}"
WEBSHOP_VAL_PATH="${WEBSHOP_VAL_PATH:-$WEBSHOP_DATA_ROOT/test.parquet}"
VAL_DUMP_DIR="${WEBSHOP_VAL_DUMP_DIR:-$PROJECT_DIR/outputs/webshop_validation/grpo}"

PROJECT_NAME="${PROJECT_NAME:-WebShop_AGENT_R1}"
EXP_NAME="${EXP_NAME:-webshop_grpo}"

python3 -m agent_r1.trainer.main_agent_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo="${AGENT_R1_NORM_ADV_BY_STD_IN_GRPO:-True}" \
    data.train_files="$WEBSHOP_TRAIN_PATH" \
    data.val_files="$WEBSHOP_VAL_PATH" \
    data.train_batch_size="$WEBSHOP_TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$WEBSHOP_MAX_PROMPT_LEN" \
    data.max_response_length="$WEBSHOP_MAX_RESPONSE_LEN" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$WEBSHOP_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$WEBSHOP_TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$WEBSHOP_LOG_PROB_MICRO_BATCH" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n="$AGENT_R1_GRPO_ROLLOUT_N" \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$CONFIG_PATH" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$WEBSHOP_LOG_PROB_MICRO_BATCH" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.rollout.agent.default_agent_flow=webshop_agent \
    actor_rollout_ref.rollout.trace.backend=mlflow \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker=5 \
    reward_model.enable=False \
    custom_reward_function.path=recipes/webshop/reward_fn.py \
    custom_reward_function.name=compute_score \
    critic.enable=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab","mlflow"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.validation_data_dir="$VAL_DUMP_DIR" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.total_epochs=10 "$@"
