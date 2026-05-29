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

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,3,4,7}
export VLLM_USE_V1=1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export HYDRA_FULL_ERROR=1
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}
export PAPER_SEARCH_BASE_URL=${PAPER_SEARCH_BASE_URL:-http://localhost:4000}
export PAPERSEARCH_SELECTOR_BASE_URL=${PAPERSEARCH_SELECTOR_BASE_URL:-http://localhost:8000}

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/recipes/paper_search/base.yaml"

PAPERSEARCH_MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
PAPERSEARCH_MAX_PROMPT_LEN=${PAPERSEARCH_MAX_PROMPT_LEN:-10240}
PAPERSEARCH_MAX_RESPONSE_LEN=${PAPERSEARCH_MAX_RESPONSE_LEN:-4096}
PAPERSEARCH_TRAIN_PATH="${PAPERSEARCH_TRAIN_PATH:-$PROJECT_DIR/data/pasa/train.parquet}"
PAPERSEARCH_VAL_PATH="${PAPERSEARCH_VAL_PATH:-$PROJECT_DIR/data/pasa/test.parquet}"
export PAPERSEARCH_SELECTOR_MODEL_NAME=${PAPERSEARCH_SELECTOR_MODEL_NAME:-selector-qwen-8b}

PROJECT_NAME=${PROJECT_NAME:-FALCON}
EXP_NAME=${EXP_NAME:-papersearch_ppo}

python3 -m agent_r1.trainer.main_agent_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$PAPERSEARCH_TRAIN_PATH" \
    data.val_files="$PAPERSEARCH_VAL_PATH" \
    data.train_batch_size=128 \
    data.max_prompt_length="$PAPERSEARCH_MAX_PROMPT_LEN" \
    data.max_response_length="$PAPERSEARCH_MAX_RESPONSE_LEN" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$PAPERSEARCH_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$CONFIG_PATH" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.rollout.agent.default_agent_flow=paper_search_agent \
    actor_rollout_ref.rollout.trace.backend=mlflow \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker=5 \
    critic.model.path="$PAPERSEARCH_MODEL_PATH" \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=16 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.enable=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab","mlflow"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.max_critic_ckpt_to_keep=3 \
    trainer.total_epochs=20 "$@"
