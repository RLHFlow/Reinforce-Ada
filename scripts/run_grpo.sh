#!/bin/bash

set -xeuo pipefail

#export VLLM_ATTENTION_BACKEND=XFORMERS
export WORKING_DIR="${PWD}"

# Model
model_name_or_path="qwen/Qwen2.5-Math-1.5B"
model_name="Qwen2.5-Math-1.5B"

# Wandb setting
project_name="Reinforce-Ada"
experiment_name="GRPO-${model_name}"

# Output
ckpts_dir="./outputs/${project_name}/${experiment_name}"
mkdir -p "${ckpts_dir}/logs"

# Trainig setting
NGPUS=8
train_prompt_bsz=512
train_prompt_mini_bsz=128

# Algorithm setting
algorithm=grpo
n=4
kl_coef=0.0
use_kl_in_reward=False
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

# Training data
train_path="./data/openr1/train.parquet"
test_path="./data/openr1/test.parquet"
train_files="['$train_path']"
test_files="['$test_path']"


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${algorithm} \
    data.train_files=${train_files} \
    data.val_files=${test_files} \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path=${model_name_or_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.default_local_dir=${ckpts_dir} \
    trainer.test_freq=50 \
    trainer.total_training_steps=400 \
    trainer.total_epochs=1000 2>&1 | tee ${ckpts_dir}/logs/log
    