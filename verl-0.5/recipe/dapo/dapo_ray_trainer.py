# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize constraint manager if enabled
        self.use_constraints = self.config.algorithm.get("use_constraints", False)
        print(f"\033[93mDAPOTrainer: use_constraints = {self.use_constraints}\033[0m")
        self.constraint_manager = None
        
        if self.use_constraints:
            from .constrained_reward_manager import ConstrainedDAPORewardManager
            
            constraint_config = self.config.algorithm.constraint_config
            self.constraint_manager = ConstrainedDAPORewardManager(
                target_length=constraint_config.target_length,
                tolerance=constraint_config.tolerance,
                lambda_init=constraint_config.lambda_init,
                lambda_lr=constraint_config.lambda_lr,
                lambda_max=constraint_config.lambda_max,
                lambda_min=constraint_config.lambda_min,
                ema_alpha=constraint_config.ema_alpha,
                momentum_beta=constraint_config.momentum_beta,
                constraint_type=constraint_config.constraint_type,
                enable_adaptive_tolerance=constraint_config.get("enable_adaptive_tolerance", False),
                adaptive_tolerance_factor=constraint_config.get("adaptive_tolerance_factor", 0.1),
                # Pass batch information for precise window calculation
                batch_size=self.config.data.train_batch_size,
                n_responses_per_prompt=self.config.actor_rollout_ref.rollout.n,
            )
            if self.constraint_manager is not None:
                print("\033[91mLoad ConstrainedDAPORewardManager Successful\033[0m")

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0

        #breakpoint()

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.gen_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                        print("Generation End")

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output
                            
                    # Check if batch is empty before assigning UIDs
                    if len(new_batch.batch) == 0:
                        print("WARNING: Empty batch encountered, skipping.")
                        progress_bar.update(1)
                        self.gen_steps += 1
                        continue

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            # Ensure the extra info arrays match the number of trajectories
                            num_trajectories = len(new_batch.non_tensor_batch["uid"])
                            #print(f"\033[94m[Reward Extra Info] Expected trajectory count: {num_trajectories}\033[0m")
                            
                            for k, v in reward_extra_infos_dict.items():
                                arr = np.array(v)
                                #print(f"\033[96m  - Metric '{k}': shape={arr.shape}, len={len(arr)}\033[0m")
                                
                                # Check if array length matches trajectory count
                                if len(arr) == num_trajectories:
                                    new_batch.non_tensor_batch[k] = arr
                                    #print(f"\033[92m    ✓ Added to non_tensor_batch\033[0m")
                                else:
                                    print(f"\033[93m    ✗ Warning: Length mismatch ({len(arr)} != {num_trajectories}). Skipping.\033[0m")

                        # Compute response_mask before applying constraints
                        if "response_mask" not in new_batch.batch:
                            new_batch.batch["response_mask"] = compute_response_mask(new_batch)

                        # Apply Lagrangian constraints if enabled
                        if self.use_constraints and self.constraint_manager is not None:
                            constraint_result = self.constraint_manager.compute_constrained_reward(
                                new_batch,
                                original_rewards=new_batch.batch["token_level_scores"],
                                return_dict=True
                            )
                            
                            # Update rewards with constrained values
                            new_batch.batch["token_level_scores"] = constraint_result["reward_tensor"]
                            
                            # Add constraint info to batch
                            if "reward_extra_info" in constraint_result:
                                num_trajectories = len(new_batch.non_tensor_batch["uid"])
                                #print(f"\033[95m[Constraint Extra Info] Expected trajectory count: {num_trajectories}\033[0m")
                                
                                for key, value in constraint_result["reward_extra_info"].items():
                                    arr = np.array(value).reshape(-1)
                                    #print(f"\033[35m  - Constraint '{key}': shape={arr.shape}, len={len(arr)}\033[0m")
                                    
                                    # For per-trajectory data (like response_lengths, constraint_violations)
                                    if key in ["response_lengths", "constraint_violations"] and len(arr) == num_trajectories:
                                        new_batch.non_tensor_batch[f"constraint_{key}"] = arr
                                        #print(f"\033[92m    ✓ Added as per-trajectory data\033[0m")
                                    # For scalar metrics, broadcast to all trajectories
                                    elif len(arr) == 1:
                                        new_batch.non_tensor_batch[f"constraint_{key}"] = np.full(num_trajectories, arr[0])
                                        #print(f"\033[92m    ✓ Broadcasted scalar to all trajectories\033[0m")
                                    else:
                                        # Skip if length doesn't match
                                        print(f"\033[91m    ✗ Unexpected shape. Skipping.\033[0m")
                            
                            # Update metrics with constraint info
                            constraint_metrics = self.constraint_manager.get_metrics()
                            print(f"\033[92mConstraint metrics: {list(constraint_metrics.keys())}\033[0m")
                            metrics.update(constraint_metrics)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        print(f"\033[36m[Filter Groups] Using metric: '{metric_name}'\033[0m")
                        
                        # Ensure we have trajectories to filter
                        if "uid" not in new_batch.non_tensor_batch or len(new_batch.non_tensor_batch["uid"]) == 0:
                            print("\033[91mWARNING: No trajectories in batch to filter, skipping.\033[0m")
                            progress_bar.update(1)
                            self.gen_steps += 1
                            continue
                        
                        # Log the shape of UIDs and metric
                        uid_count = len(new_batch.non_tensor_batch["uid"])
                        print(f"\033[36m  - UID count: {uid_count}\033[0m")
                        
                        # Check if metric exists in non_tensor_batch
                        if metric_name not in ["seq_final_reward", "seq_reward"] and metric_name not in new_batch.non_tensor_batch:
                            print(f"\033[91mERROR: Metric '{metric_name}' not found in non_tensor_batch. Available keys: {list(new_batch.non_tensor_batch.keys())}\033[0m")
                            progress_bar.update(1)
                            self.gen_steps += 1
                            continue
                            
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        
                        # Log metric shape before filtering
                        if metric_name in new_batch.non_tensor_batch:
                            metric_shape = np.array(new_batch.non_tensor_batch[metric_name]).shape
                            print(f"\033[36m  - Metric '{metric_name}' shape: {metric_shape}\033[0m")
                        
                        try:
                            for uid, metric_val in zip(
                                new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                            ):
                                prompt_uid2metric_vals[uid].append(metric_val)
                        except ValueError as e:
                            print(f"\033[91mERROR in zip: {e}\033[0m")
                            print(f"\033[91m  - UID length: {len(new_batch.non_tensor_batch['uid'])}\033[0m")
                            print(f"\033[91m  - Metric length: {len(new_batch.non_tensor_batch[metric_name])}\033[0m")
                            raise

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)
                        
                        print(f"\033[36m  - Kept trajectories: {len(kept_traj_idxs)} out of {len(new_batch.non_tensor_batch['uid'])}\033[0m")
                        print(f"\033[36m  - Kept prompt UIDs: {len(kept_prompt_uids)} unique prompts\033[0m")

                        # Only filter if we have trajectories to keep
                        if kept_traj_idxs:
                            print(f"\033[32m  - Filtering batch with indices: min={min(kept_traj_idxs)}, max={max(kept_traj_idxs)}\033[0m")
                            new_batch = new_batch[kept_traj_idxs]
                            batch = new_batch if batch is None else DataProto.concat([batch, new_batch])
                        else:
                            print(f"\033[93mWARNING: No trajectories kept after filtering (all have zero std). Skipping batch.\033[0m")

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                self.gen_steps += 1
                                is_last_step = self.gen_steps >= self.total_training_steps
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
