#!/bin/bash

set -x
export WORKING_DIR="${PWD}"

# Set XFormers backend to avoid CUDA errors
#export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH=/mnt/nushare2/data/baliao/PLLMs/qwen/Qwen2.5-7B-Instruct
model_name=Qwen2.5-7B-Instruct
train_path=/mnt/nushare2/data/baliao/hint/data/final/luffy/train15k.parquet
test_path=/mnt/nushare2/data/baliao/dynamic_filter/data/test/test.parquet

project_name="self-rl-hint"
experiment_name="LUFFY_${model_name}"

ckpts_dir="/mnt/nushare2/data/baliao/hint/06_baselines/${experiment_name}"
mkdir -p "${ckpts_dir}/logs"
chown -R 110541254:110541254 ${ckpts_dir}

export WANDB_API_KEY="9f81cffd97cee8ca6dd3949f56beb6f87e223cc3"
export WANDB_ENTITY="baliao-uva"
export WANDB_MODE="offline"
export WANDB_DIR=${ckpts_dir}/logs


# Train over a single node, 8 A100-80GB GPUs.
python3 -m verl.mix_src.main_mix_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${train_path} \
    data.val_files=${test_path} \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.max_prefix_len=8192 \
    algorithm.kl_ctrl.kl_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.default_local_dir=${ckpts_dir} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_sft_prefix_reward=False \
    actor_rollout_ref.rollout.prefix_share_across_samples=False \
    actor_rollout_ref.rollout.prefix_strategy=random \
    actor_rollout_ref.rollout.n_prefix=1 \
    actor_rollout_ref.rollout.min_prefix_ratio=1.0 \
    actor_rollout_ref.rollout.max_prefix_ratio=1.0 \
    actor_rollout_ref.rollout.prefix_reward_weight_alpha=1.0 \
    actor_rollout_ref.ref.use_ref=False \
    actor_rollout_ref.actor.use_off_policy_loss=True \
    actor_rollout_ref.actor.off_policy_normalize=False \
    actor_rollout_ref.actor.off_policy_reshape="p_div_p_0.1" \
    actor_rollout_ref.actor.off_policy_loss_impl=token \
    algorithm.grpo_use_std=False \
    actor_rollout_ref.actor.loss_remove_token_mean=True \
    actor_rollout_ref.actor.loss_remove_clip=True \
    data.reward_impl_version=3 \
    trainer.max_optim_to_keep=2 \
    data.shuffle=True \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=30 \
    trainer.total_training_steps=500 \
    trainer.resume_mode="auto" 2>&1 | tee ${ckpts_dir}/logs/log