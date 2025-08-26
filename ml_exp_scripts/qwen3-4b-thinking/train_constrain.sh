#!/usr/bin/env bash
# Fix vLLM and transformers compatibility issue

project_name='Constrained-DAPO'
exp_name='Constrained-DAPO-Qwen3-4B-Thinking-32k-to-8k'

adv_estimator=grpo

# KL settings (same as original DAPO)
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# Length settings - targeting 4k from 10k
max_prompt_length=1024
# Increased to 32k response with 64k context window
max_response_length=32768

enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

# Adjusted batch sizes for single node 8 GPUs
train_prompt_bsz=64 # Reduced from 512 to 64 for single node
gen_prompt_bsz=64
val_prompt_bsz=64
n_resp_per_prompt=8
train_prompt_mini_bsz=32
filter_groups_enable=False
filter_groups_metric=acc
max_num_gen_batches=5

# Paths
MODEL_PATH=${MODEL_PATH:-"/jizhicfs/hymiezhao/models/Qwen3-4B-Thinking-2507"}
CKPTS_DIR=${CKPTS_DIR:-"./${project_name}/${exp_name}"}
TRAIN_FILES="['./dataset/train_data/4k_high_quality_deepmath.parquet']"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1
val_temperature=0.6
val_top_p=0.95
val_top_k=-1

# Performance Related Parameter - Adjusted for single node
sp_size=1
actor_ppo_max_token_len=34816
infer_ppo_max_token_len=48000
offload=True
fsdp_size=-1

# Constraint settings for 10k->4k reduction (ratio-based)
use_constraints=True
target_length=8192
tolerance=0.125  # 12.5% deviation allowed
lambda_init=0.01  # Start with small penalty
lambda_lr=0.02    # Conservative learning rate for gradual adaptation  
lambda_max=2.0    # Max penalty matches reward scale
constraint_type="average"

aime24=./dataset/test_data/valid.aime24.parquet
aime25=./dataset/test_data/valid.aime25.parquet
hmmt25=./dataset/test_data/valid.hmmt25.parquet
gpqa=./dataset/test_data/valid.gpqa.parquet
olympiad_bench=./dataset/test_data/valid.olympiad_bench.parquet

# actor_rollout_ref.rollout.max_num_seqs
# actor_rollout_ref.rollout.max_num_seqs=128 \

python -m recipe.constraint.main_dapo \
    data.train_files="${TRAIN_FILES}" \
    data.val_files='['$aime24','$aime25','$hmmt25']' \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=${val_prompt_bsz} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.use_constraints=${use_constraints} \
    algorithm.constraint_config.target_length=${target_length} \
    algorithm.constraint_config.tolerance=${tolerance} \
    algorithm.constraint_config.lambda_init=${lambda_init} \
    algorithm.constraint_config.lambda_lr=${lambda_lr} \
    algorithm.constraint_config.lambda_max=${lambda_max} \
    algorithm.constraint_config.constraint_type=${constraint_type} \
    algorithm.filter_groups.enable=${filter_groups_enable} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    custom_reward_function.path="./utils/reward_utils/my_reward_func.py" \
    custom_reward_function.name="my_reward_func" \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=8192 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.total_epochs=5 \
    trainer.total_training_steps=200 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10
