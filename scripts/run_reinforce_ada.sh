#!/bin/bash

set -xeuo pipefail

export VLLM_ATTENTION_BACKEND=XFORMERS
export WORKING_DIR="${PWD}"

# Model
model_name_or_path=qwen/Qwen2.5-Math-1.5B
model_name=Qwen2.5-Math-1.5B

# Wandb setting
project_name=Reinforce-Ada
exp_name=Reinforce-Ada_${model_name}

# Output
ckpts_dir="./outputs/${project_name}/${exp_name}"
mkdir -p "${ckpts_dir}/logs"

# Trainig setting
NGPUS=8
train_prompt_bsz=512
train_prompt_mini_bsz=128

sp_size=1
tp_size=1
use_dynamic_bsz=True
offload=False

max_prompt_length=$((1024 * 1))
max_response_length=$((2048))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 2))
overlong_penalty_factor=0.0
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))

loss_agg_mode="token-mean"
temperature=1.0
top_p=1.0
top_k=-1
val_temperature=0.0
val_top_p=1.0

# Algorithm setting
adv_estimator=grpo
n=4
kl_coef=0.0
use_kl_in_reward=False
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

## Reinforce-Ada setting
multiround_adaptive_downsampling=True
reinforce_ada_choice="balanced" # "positive_focused" or "balanced"
global_stat_est=True
norm_adv_by_std_in_grpo=False

# Training data
train_path=./data/openr1/train.parquet
test_path=./data/openr1/test.parquet
train_files="['$train_path']"
test_files="['$test_path']"


python3 -m verl.trainer.main_ppo \
    data.train_files=${train_files} \
    data.val_files=${test_files} \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n} \
    algorithm.multiround_adaptive_downsampling=${multiround_adaptive_downsampling} \
    algorithm.reinforce_ada_choice=${reinforce_ada_choice} \
    algorithm.global_stat_est=${global_stat_est} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.model.path="${model_name_or_path}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0. \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tp_size} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=TRUE \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${NGPUS} \
    reward_model.reward_manager=naive \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.resume_mode="auto" \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS}" \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=50 \
    trainer.save_freq=50 \
    trainer.total_epochs=1000 \
    trainer.total_training_steps=400 \
    trainer.default_local_dir=${ckpts_dir} \
    trainer.log_val_generations=10 2>&1 | tee ${ckpts_dir}/logs/log                                                                  