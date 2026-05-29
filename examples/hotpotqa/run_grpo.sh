set -x

# 4 GPUs: training + vLLM use 0–3; each of the 4 agent workers runs BGE on its own GPU (cuda:0..3).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HOTPOTQA_EMBEDDING_PER_WORKER_GPU=${HOTPOTQA_EMBEDDING_PER_WORKER_GPU:-1}
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://172.17.0.1:5000}

# GRPO: multiple rollouts per task for group-relative advantages (verl rollout.n).
AGENT_R1_GRPO_ROLLOUT_N="${AGENT_R1_GRPO_ROLLOUT_N:-8}"
# Keep the effective unique-task budget near 256: train_batch_size * rollout.n ~= 256.
HOTPOTQA_GRPO_BASE_TRAIN_BATCH="${HOTPOTQA_GRPO_BASE_TRAIN_BATCH:-256}"
HOTPOTQA_GRPO_BASE_LOG_PROB_MICRO_BATCH="${HOTPOTQA_GRPO_BASE_LOG_PROB_MICRO_BATCH:-8}"
HOTPOTQA_TRAIN_BATCH_SIZE="$((HOTPOTQA_GRPO_BASE_TRAIN_BATCH / AGENT_R1_GRPO_ROLLOUT_N))"
HOTPOTQA_LOG_PROB_MICRO_BATCH="$((HOTPOTQA_GRPO_BASE_LOG_PROB_MICRO_BATCH / AGENT_R1_GRPO_ROLLOUT_N))"
if [[ "$HOTPOTQA_TRAIN_BATCH_SIZE" -lt 1 ]]; then
    echo "❌ HOTPOTQA_GRPO_BASE_TRAIN_BATCH ($HOTPOTQA_GRPO_BASE_TRAIN_BATCH) must be >= AGENT_R1_GRPO_ROLLOUT_N ($AGENT_R1_GRPO_ROLLOUT_N)." >&2
    exit 1
fi
if [[ "$HOTPOTQA_LOG_PROB_MICRO_BATCH" -lt 1 ]]; then
    HOTPOTQA_LOG_PROB_MICRO_BATCH=1
fi

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/recipes/hotpotqa/base.yaml"

export HOTPOTQA_DATA_ROOT="${HOTPOTQA_DATA_ROOT:-$PROJECT_DIR/data/corpus/hotpotqa}"

HOTPOTQA_MODEL_PATH=${HOTPOTQA_MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}

# Length budget (vs. Agent-R1 `run_ppo_hotpotqa.sh`; semantics differ):
# - Legacy: multi-turn tokens concatenated into one trajectory → data.max_prompt_length=8192, full response=8192,
#   per-turn max_response_length_single_turn=1024.
# - This script (AGENT_R1 AgentFlow): each step rebuilds the prompt + one generate per step; the user block adds a
#   "Recent tool / format issues" section, so we use a larger prompt budget; per-step response matches legacy 1024
#   to reduce <tool_call> JSON truncation at max_tokens.
HOTPOTQA_MAX_PROMPT_LEN=${HOTPOTQA_MAX_PROMPT_LEN:-10240}
HOTPOTQA_MAX_RESPONSE_LEN=${HOTPOTQA_MAX_RESPONSE_LEN:-1024}

TRAIN_PATH="$PROJECT_DIR/data/corpus/hotpotqa/train.parquet"
VAL_PATH="$PROJECT_DIR/data/corpus/hotpotqa/validation.parquet"
HOTPOTQA_WIKI2_VAL_PATH="${HOTPOTQA_WIKI2_VAL_PATH:-$PROJECT_DIR/data/corpus/hotpotqa/2wikimultihopqa_validation.parquet}"
HOTPOTQA_MUSIQUE_VAL_PATH="${HOTPOTQA_MUSIQUE_VAL_PATH:-$PROJECT_DIR/data/corpus/hotpotqa/musique_validation.parquet}"
HOTPOTQA_ENABLE_CROSS_EVAL="${HOTPOTQA_ENABLE_CROSS_EVAL:-auto}"

build_val_files() {
    local files=("$VAL_PATH")
    local include_missing=false
    if [[ "$HOTPOTQA_ENABLE_CROSS_EVAL" == "1" || "$HOTPOTQA_ENABLE_CROSS_EVAL" == "true" ]]; then
        include_missing=true
    fi
    if [[ "$HOTPOTQA_ENABLE_CROSS_EVAL" != "0" && "$HOTPOTQA_ENABLE_CROSS_EVAL" != "false" ]]; then
        if [[ -f "$HOTPOTQA_WIKI2_VAL_PATH" || "$include_missing" == true ]]; then
            files+=("$HOTPOTQA_WIKI2_VAL_PATH")
        fi
        if [[ -f "$HOTPOTQA_MUSIQUE_VAL_PATH" || "$include_missing" == true ]]; then
            files+=("$HOTPOTQA_MUSIQUE_VAL_PATH")
        fi
    fi

    local val_files="["
    local path
    for path in "${files[@]}"; do
        val_files+="\"$path\","
    done
    val_files="${val_files%,}]"
    echo "$val_files"
}

VAL_FILES="$(build_val_files)"

PROJECT_NAME='HotpotQA_AGENT_R1'
EXP_NAME="${EXP_NAME:-hotpotqa_grpo}"

python3 -m agent_r1.trainer.main_agent_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo="${AGENT_R1_NORM_ADV_BY_STD_IN_GRPO:-True}" \
    data.train_files="$TRAIN_PATH" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size="$HOTPOTQA_TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$HOTPOTQA_MAX_PROMPT_LEN" \
    data.max_response_length="$HOTPOTQA_MAX_RESPONSE_LEN" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$HOTPOTQA_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$HOTPOTQA_TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$HOTPOTQA_LOG_PROB_MICRO_BATCH" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n="$AGENT_R1_GRPO_ROLLOUT_N" \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$CONFIG_PATH" \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.rollout.agent.default_agent_flow=hotpotqa_agent \
    actor_rollout_ref.rollout.trace.backend=mlflow \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker=5 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.enable=False \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.99 \
    reward_model.enable=False \
    custom_reward_function.path=recipes/hotpotqa/reward_fn.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab","mlflow"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.total_epochs=5 "$@"
