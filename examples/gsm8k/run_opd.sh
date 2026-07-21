#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

if [[ -n "${VERL_ROOT:-}" ]]; then
  export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

"$PYTHON_BIN" - <<'PY'
try:
    from verl.experimental.teacher_loop import MultiTeacherModelManager  # noqa: F401
    from verl.trainer.distillation import distillation_ppo_loss  # noqa: F401
    from verl.trainer.ppo.utils import Role
except (ImportError, AttributeError) as exc:
    raise SystemExit(
        "OPD-enabled verl is required. Set VERL_ROOT to a verl source checkout "
        "that contains trainer/distillation and experimental/teacher_loop."
    ) from exc
if not hasattr(Role, "TeacherModel"):
    raise SystemExit("The selected verl checkout does not define Role.TeacherModel.")
PY

STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-1.7B}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
GSM8K_TRAIN_PATH="${GSM8K_TRAIN_PATH:-$HOME/data/gsm8k/train.parquet}"
GSM8K_VAL_PATH="${GSM8K_VAL_PATH:-$HOME/data/gsm8k/test.parquet}"
MAX_PROMPT_LEN="${GSM8K_MAX_PROMPT_LEN:-2048}"
MAX_RESPONSE_LEN="${GSM8K_MAX_RESPONSE_LEN:-1024}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}"
STUDENT_GPUS_PER_NODE="${STUDENT_GPUS_PER_NODE:-2}"
TEACHER_GPUS_PER_NODE="${TEACHER_GPUS_PER_NODE:-2}"
AGENT_FLOW_WORKERS="${AGENT_FLOW_WORKERS:-4}"
PROJECT_NAME="${PROJECT_NAME:-AGENT_R1_OPD}"
EXP_NAME="${EXP_NAME:-gsm8k_opd_qwen3_1p7b_teacher4b_k1}"

"$PYTHON_BIN" -m agent_r1.trainer.main_agent_ppo \
  trainer.use_legacy_worker_impl=disable \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$GSM8K_TRAIN_PATH" \
  data.val_files="$GSM8K_VAL_PATH" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LEN" \
  data.max_response_length="$MAX_RESPONSE_LEN" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$STUDENT_MODEL" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN + 1)) \
  actor_rollout_ref.rollout.prompt_length="$MAX_PROMPT_LEN" \
  actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LEN" \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.agent.num_workers="$AGENT_FLOW_WORKERS" \
  actor_rollout_ref.rollout.agent.default_agent_flow=single_step_agent \
  actor_rollout_ref.rollout.trace.backend=null \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  critic.enable=False \
  reward_model.enable=False \
  custom_reward_function.path=recipes/gsm8k/reward_fn.py \
  custom_reward_function.name=compute_score \
  distillation.enabled=True \
  distillation.n_gpus_per_node="$TEACHER_GPUS_PER_NODE" \
  distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL" \
  distillation.teacher_models.teacher_model.inference.name=vllm \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.45 \
  distillation.teacher_models.teacher_model.inference.max_model_len=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN + 1)) \
  distillation.distillation_loss.loss_mode=k1 \
  distillation.distillation_loss.use_policy_gradient=True \
  distillation.distillation_loss.use_task_rewards=False \
  distillation.distillation_loss.distillation_loss_coef=1.0 \
  distillation.distillation_loss.loss_max_clamp=10.0 \
  distillation.distillation_loss.log_prob_min_clamp=-10.0 \
  trainer.logger='["console"]' \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXP_NAME" \
  trainer.n_gpus_per_node="$STUDENT_GPUS_PER_NODE" \
  trainer.nnodes=1 \
  trainer.val_before_train=True \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.max_actor_ckpt_to_keep=3 \
  trainer.total_epochs=1 "$@"
