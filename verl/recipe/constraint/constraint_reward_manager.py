import torch
import numpy as np
import os
from typing import Dict, Optional, Union
from verl import DataProto


class ConstraintRewardManager:
    """
    Manages Lagrangian constraints for DAPO, specifically for controlling output length.
    
    Mathematical formulation (ratio-based):
    $$\mathcal{L}(\theta, \lambda) = -\mathbb{E}_{\pi_\theta}[R(x,y)] + \lambda \cdot g(\pi_\theta)$$
    
    where $g(\pi_\theta) = \frac{\mathbb{E}_{\pi_\theta}[|y|]}{L_{target}} - 1$ is the normalized constraint.
    
    The augmented reward becomes:
    $$\tilde{R}(x,y) = R(x,y) - \lambda \cdot (\frac{|y|}{L_{target}} - 1)$$
    """
    
    def __init__(
        self,
        target_length: int = 4096,
        lambda_init: float = 0.01,  # Adjusted for ratio-based constraint
        lambda_lr: float = 0.02,  # Conservative learning rate for stable convergence
        lambda_max: float = 2.0,  # Reduced from 10.0 to match reward scale
        lambda_min: float = 0.0,
        constraint_type: str = "average",  # "average" or "max"
    ):
        self.target_length = target_length
        self.lambda_lr = lambda_lr
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.constraint_type = constraint_type
        
        # Lagrange multiplier
        self.lagrange_multiplier = lambda_init
        
        # Tracking statistics
        self.stats = {
            "avg_length": [],
            "max_length": [],
            "min_length": [],
            "std_length": [],
            "p95_length": [],
            "constraint_violation": [],
            "lagrange_multiplier": [],
            "num_violations": [],
            "penalty_magnitude": [],
            "satisfaction_rate": [],
            "penalty_active_rate": [],
            "avg_active_penalty": [],
            "lambda_change_rate": [],
            # Training accuracy metrics
            "training_acc": [],
        }

    def compute_accuracy(self, original_rewards: torch.Tensor, response_lengths: list) -> float:
        """
        Compute training accuracy from original rewards.
        +1 reward means correct, -1 reward means incorrect.

        Args:
            original_rewards: Original reward tensor before constraint modification
            response_lengths: List of response lengths for each sample

        Returns:
            Accuracy (fraction of correct samples)
        """
        correct_count = 0
        total_count = 0

        for i in range(len(original_rewards)):
            if response_lengths[i] > 0:
                # Get the final reward (at the last response token)
                final_reward = original_rewards[i, response_lengths[i] - 1].item()
                # +1 means correct
                if final_reward > 0:
                    correct_count += 1
                total_count += 1

        return correct_count / total_count if total_count > 0 else 0.0

    def compute_constrained_reward(
        self, 
        batch: DataProto,
        original_rewards: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, Dict]]]:
        """
        Apply Lagrangian constraint to rewards.
        
        Formulas (ratio-based):
        - Augmented reward: $\tilde{r}_t = r_t - \lambda \cdot \mathbb{1}[t = T] \cdot (\frac{T}{L_{target}} - 1)$
        - Constraint violation: $g = \frac{\mathbb{E}[|y|]}{L_{target}} - 1$
        - Lambda update: $\lambda = \text{clip}(\lambda + \eta_\lambda \cdot g, \lambda_{min}, \lambda_{max})$
        """
        # Get rewards
        if original_rewards is None:
            rewards = batch.batch.get("token_level_scores", batch.batch.get("token_level_rewards"))
        else:
            rewards = original_rewards
            
        if rewards is None:
            raise ValueError("No rewards found in batch")

        # Clone to avoid in-place modification
        reward_tensor = rewards.clone()
        
        # Get response masks and compute lengths
        if "response_mask" not in batch.batch:
            raise ValueError("response_mask not found in batch. Please compute it before calling this function.")
        response_mask = batch.batch["response_mask"]
        
        # Compute actual response lengths
        response_lengths = []
        constraint_violations = []
        num_violations = 0
        total_penalty = 0.0
        
        # Get the actual sequence length from reward tensor
        seq_len = reward_tensor.shape[1]
        
        for i in range(len(reward_tensor)):
            # Find actual response length
            mask = response_mask[i]
            
            # response_mask should have the same length as reward_tensor
            if len(mask) != seq_len:
                raise ValueError(
                    f"Response mask length ({len(mask)}) doesn't match reward tensor length ({seq_len}). "
                    f"This should not happen if response_mask is computed correctly."
                )
                    
            valid_response_indices = torch.where(mask > 0)[0]
            if len(valid_response_indices) > 0:
                valid_response_length = len(valid_response_indices)
            else:
                valid_response_length = 0
            
            # Additional safety check
            if valid_response_length > seq_len:
                print(f"\033[93m[WARNING] Computed response length ({valid_response_length}) exceeds reward tensor length ({seq_len}). "
                      f"Clamping to tensor bounds.\033[0m")
                valid_response_length = seq_len
                    
            response_lengths.append(valid_response_length)
            
            # Compute constraint violation for this sample (ratio-based)
            violation_ratio = valid_response_length / self.target_length - 1.0
            constraint_violations.append(violation_ratio)
            
            if violation_ratio > 0:  # Any violation above target
                num_violations += 1
            
            # Apply penalty at the last token of the response
            # Formula: $\tilde{r}_T = r_T - \lambda \cdot (\frac{|y|}{L_{target}} - 1)$
            if valid_response_length > 0:
                # penalty = self.lagrange_multiplier * violation_ratio
                penalty = max(0, self.lagrange_multiplier * violation_ratio)
                reward_tensor[i, valid_response_length - 1] -= penalty
                # total_penalty += abs(penalty)
                total_penalty += penalty
                
        # Clip reward tensor to prevent extreme values
        # This helps maintain training stability
        reward_tensor = torch.clamp(reward_tensor, min=-1.0, max=1.0)
                
        # Compute constraint statistics
        avg_length = np.mean(response_lengths)
        max_length = np.max(response_lengths)
        
        # Compute constraint violation based on type (ratio-based)
        if self.constraint_type == "average":
            # Average constraint: $g = \frac{\mathbb{E}[|y|]}{L_{target}} - 1$
            current_constraint_violation = avg_length / self.target_length - 1.0
        else:  # "max"
            # Max constraint: $g = \frac{\max(|y|)}{L_{target}} - 1$
            current_constraint_violation = max_length / self.target_length - 1.0
            
        # Update Lagrange multiplier directly based on current constraint violation
        # Formula: $\lambda^{(t+1)} = \text{clip}(\lambda^{(t)} + \eta_\lambda \cdot g^{(t)}, \lambda_{min}, \lambda_{max})$
        self.lagrange_multiplier += self.lambda_lr * current_constraint_violation
            
        # Clip lambda to valid range
        self.lagrange_multiplier = np.clip(
            self.lagrange_multiplier, self.lambda_min, self.lambda_max
        )

        # Compute training accuracy from original rewards
        current_accuracy = self.compute_accuracy(rewards, response_lengths)

        # Update statistics
        self.stats["avg_length"].append(avg_length)
        self.stats["max_length"].append(max_length)
        self.stats["min_length"].append(np.min(response_lengths))
        self.stats["std_length"].append(np.std(response_lengths))
        self.stats["p95_length"].append(np.percentile(response_lengths, 95))
        self.stats["constraint_violation"].append(current_constraint_violation)
        self.stats["lagrange_multiplier"].append(self.lagrange_multiplier)
        self.stats["num_violations"].append(num_violations / len(response_lengths))
        self.stats["penalty_magnitude"].append(total_penalty / len(response_lengths))
        
        # Satisfaction rate (within target length)
        within_target = sum(1 for l in response_lengths if l <= self.target_length)
        self.stats["satisfaction_rate"].append(within_target / len(response_lengths))
        
        # Penalty statistics (now using ratio-based violations)
        penalties = [self.lagrange_multiplier * v for v in constraint_violations]
        non_zero_penalties = [p for p in penalties if abs(p) > 1e-6]
        if non_zero_penalties:
            self.stats["penalty_active_rate"].append(len(non_zero_penalties) / len(penalties))
            self.stats["avg_active_penalty"].append(np.mean(np.abs(non_zero_penalties)))
        else:
            self.stats["penalty_active_rate"].append(0.0)
            self.stats["avg_active_penalty"].append(0.0)
            
        # Lambda change rate
        if len(self.stats["lagrange_multiplier"]) > 1:
            self.stats["lambda_change_rate"].append(
                (self.stats["lagrange_multiplier"][-1] - self.stats["lagrange_multiplier"][-2]) / self.lambda_lr
            )
        else:
            self.stats["lambda_change_rate"].append(0.0)

        # Update accuracy statistics
        self.stats["training_acc"].append(current_accuracy)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    "response_lengths": response_lengths,
                    "constraint_violations": constraint_violations,
                    "avg_length": avg_length,
                    "max_length": max_length,
                    "lagrange_multiplier": self.lagrange_multiplier,
                    "num_violations": num_violations,
                    "penalty_magnitude": total_penalty / len(response_lengths),
                }
            }
        else:
            return reward_tensor
            
    def get_metrics(self) -> Dict[str, float]:
        """Get current constraint optimization metrics."""
        metrics = {}
        
        # Use the last recorded values from stats
        if self.stats["avg_length"]:
            metrics["constraint/avg_length"] = self.stats["avg_length"][-1]
            metrics["constraint/std_length"] = self.stats["std_length"][-1]
            metrics["constraint/min_length"] = self.stats["min_length"][-1]
            metrics["constraint/max_length"] = self.stats["max_length"][-1]
            metrics["constraint/p95_length"] = self.stats["p95_length"][-1]
            
            # Constraint satisfaction metrics
            metrics["constraint/satisfaction_rate"] = self.stats["satisfaction_rate"][-1]
            metrics["constraint/num_violations"] = self.stats["num_violations"][-1]
            
            # Distance from target (relative)
            if self.stats["avg_length"][-1] > 0:
                metrics["constraint/avg_relative_distance"] = abs(self.stats["avg_length"][-1] / self.target_length - 1.0)
            
        # Penalty statistics
        if self.stats["penalty_magnitude"]:
            metrics["constraint/avg_penalty"] = self.stats["penalty_magnitude"][-1]
            metrics["constraint/penalty_active_rate"] = self.stats["penalty_active_rate"][-1]
            if self.stats["avg_active_penalty"]:
                metrics["constraint/avg_active_penalty"] = self.stats["avg_active_penalty"][-1]
                    
        # Lambda tracking
        if self.stats["lagrange_multiplier"]:
            metrics["constraint/lambda"] = self.stats["lagrange_multiplier"][-1]
            if self.stats["lambda_change_rate"]:
                metrics["constraint/lambda_change_rate"] = self.stats["lambda_change_rate"][-1]
                
        # Violation statistics
        if self.stats["constraint_violation"]:
            metrics["constraint/avg_violation_ratio"] = self.stats["constraint_violation"][-1]
            
        # Add current values
        metrics["constraint/target_length"] = self.target_length

        # Training accuracy metrics
        if self.stats["training_acc"]:
            metrics["constraint/training_acc"] = self.stats["training_acc"][-1]

        return metrics
        
    def reset_stats(self):
        """Reset tracking statistics."""
        for key in self.stats:
            self.stats[key] = []
        
    def get_summary_string(self) -> str:
        """Get a human-readable summary of current metrics."""
        metrics = self.get_metrics()
        
        summary_parts = []
        
        if "constraint/avg_length" in metrics:
            summary_parts.append(
                f"Avg Length: {metrics['constraint/avg_length']:.1f} "
                f"(target: {self.target_length}, "
                f"sat_rate: {metrics.get('constraint/satisfaction_rate', 0):.2%})"
            )
            
        if "constraint/lambda" in metrics:
            summary_parts.append(f"λ: {metrics['constraint/lambda']:.4f}")
            
        if "constraint/avg_penalty" in metrics:
            summary_parts.append(
                f"Penalty: {metrics['constraint/avg_penalty']:.3f} "
                f"(active: {metrics.get('constraint/penalty_active_rate', 0):.2%})"
            )

        if "constraint/training_acc" in metrics:
            summary_parts.append(f"Acc: {metrics['constraint/training_acc']:.2%}")

        return " | ".join(summary_parts)


    def save_state(self, file_path: str):
        """
        Saves the current state of the manager to a file for checkpointing.

        Args:
            file_path (str): The path to the file where the state will be saved.
        """
        # Ensure the directory exists
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        state_dict = {
            'lagrange_multiplier': self.lagrange_multiplier,
            'stats': self.stats,  # Save full stats for continuous logging
        }
        
        try:
            torch.save(state_dict, file_path)
            print(f"\033[92m✅ ConstraintRewardManager state saved to {file_path}\033[0m")
        except Exception as e:
            print(f"\033[91m❌ Error saving ConstraintRewardManager state: {e}\033[0m")

    def load_state(self, file_path: str):
        """
        Loads the state of the manager from a file to resume training.

        Args:
            file_path (str): The path to the file from which to load the state.
        """
        if not os.path.exists(file_path):
            print(f"\033[93m⚠️ [WARNING] Checkpoint file not found at {file_path}. Starting with a fresh state.\033[0m")
            exit()

        try:
            state_dict = torch.load(file_path, map_location='cpu', weights_only=False)
            
            # Restore core state variables using .get() for safety
            self.lagrange_multiplier = state_dict.get('lagrange_multiplier', self.lagrange_multiplier)
            
            # Restore stats dictionary for complete historical data
            loaded_stats = state_dict.get('stats', {})
            for key in self.stats:
                if key in loaded_stats:
                    self.stats[key] = loaded_stats[key]

            print(f"\033[92m✅ ConstraintRewardManager state loaded from {file_path}\033[0m")
            print(f"  - Resumed λ: {self.lagrange_multiplier:.4f}")

        except Exception as e:
            print(f"\033[91m❌ Error loading ConstraintRewardManager state from {file_path}: {e}. \033[0m")
            raise RuntimeError(f"Failed to load ConstraintRewardManager state from {file_path}: {e}")
        