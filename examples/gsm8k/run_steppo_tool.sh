#!/usr/bin/env bash
set -x

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_V1=1
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_HOME=/usr/local/cuda

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/recipes/gsm8k/base.yaml"
EXP_NAME="${EXP_NAME:-gsm8k_steppo_tool}"

python3 -m agent_r1.trainer.main_agent_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$HOME/data/gsm8k_tool/train.parquet \
    data.val_files=$HOME/data/gsm8k_tool/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.prompt_length=4096 \
    actor_rollout_ref.rollout.response_length=2048 \
    actor_rollout_ref.rollout.agent.agent_flow_config_path="$CONFIG_PATH" \
    actor_rollout_ref.rollout.agent.default_agent_flow=gsm8k_tool \
    actor_rollout_ref.rollout.agent.max_steps=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    custom_reward_function.path=recipes/gsm8k/reward_fn.py \
    custom_reward_function.name=compute_score \
    trainer.project_name='agent_r1_gsm8k_steppo_tool' \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
