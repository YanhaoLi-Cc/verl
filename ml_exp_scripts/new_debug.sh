#!/usr/bin/env bash
# Fix vLLM and transformers compatibility issue


#Found existing installation: protobuf 5.26.1   
#Found existing installation: tensordict 0.9.1

export VLLM_ATTENTION_BACKEND=XFORMERS

ray stop
CUDA_VISIBLE_DEVICES=4 ray start --head --include-dashboard=true --num-cpus=50 --num-gpus=1

project_name='Constrained-DAPO-0820'
exp_name='Constrained-DAPO-Qwen3-4B-10k-to-4k-ratio'

adv_estimator=grpo

# KL settings (same as original DAPO)
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# Length settings - targeting 4k from 10k
max_prompt_length=512
# Increased to 32k response with 64k context window
max_response_length=512
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

# Adjusted batch sizes for single node 8 GPUs
train_prompt_bsz=4 # Reduced from 512 to 64 for single node
val_prompt_bsz=8
n_resp_per_prompt=4
train_prompt_mini_bsz=4  # Reduced from 32 to 8
filter_groups_enable=False
filter_groups_metric=acc
max_num_gen_batches=4
gen_prompt_bsz=$((train_prompt_bsz * 1))

# Ray
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Paths
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
RAY_DATA_HOME=${RAY_DATA_HOME:-"/data/malu/verl_long2short/verl"}
MODEL_PATH=${MODEL_PATH:-"/data/malu/Qwen2.5-0.5B-Instruct"}
CKPTS_DIR=${CKPTS_DIR:-"/data/malu/checkpoint/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"/data/malu/ReLIFT/dataset/train_data/sub_openr1.parquet"}
TEST_FILE=${TEST_FILE:-"/data/malu/ReLIFT/dataset/train_data/sampled_valid.parquet"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

# Performance Related Parameter - Adjusted for single node
sp_size=1  # Reduced from 4 to 2 for single node
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
echo $actor_ppo_max_token_len
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=True
gen_tp=1
fsdp_size=-1

# Constraint settings for 10k->4k reduction (ratio-based)
use_constraints=True
target_length=256
tolerance=0.125  # 12.5% deviation allowed
lambda_init=0.01  # Start with small penalty
lambda_lr=0.02    # Conservative learning rate for gradual adaptation  
lambda_max=2.0    # Max penalty matches reward scale
constraint_type="average"

CUDA_VISIBLE_DEVICES=4 python -m recipe.constraint.main_dapo \
    custom_reward_function.path="./utils/reward_utils/my_reward_func.py" \
    custom_reward_function.name="my_reward_func" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=${val_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
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
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=65536 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    trainer.save_freq=-1 \
    trainer.total_epochs=3 \
    trainer.total_training_steps=200 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10