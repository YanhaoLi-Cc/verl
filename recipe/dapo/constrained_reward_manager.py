import torch
import numpy as np
from typing import Dict, Optional, Union
from verl import DataProto


class ConstrainedDAPORewardManager:
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
        tolerance: float = 0.125,  # Now a ratio: 0.125 means 12.5% deviation allowed
        lambda_init: float = 0.01,  # Adjusted for ratio-based constraint
        lambda_lr: float = 0.02,  # Conservative learning rate for stable convergence
        lambda_max: float = 2.0,  # Reduced from 10.0 to match reward scale
        lambda_min: float = 0.0,
        ema_alpha: float = 0.95,
        momentum_beta: float = 0.9,
        constraint_type: str = "average",  # "average" or "max"
        enable_adaptive_tolerance: bool = False,
        adaptive_tolerance_factor: float = 0.1,
        batch_size: Optional[int] = None,  # Batch size for dynamic window calculation
        n_responses_per_prompt: Optional[int] = None,  # Number of responses per prompt
    ):
        self.target_length = target_length
        self.tolerance = tolerance
        self.lambda_lr = lambda_lr
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.ema_alpha = ema_alpha
        self.momentum_beta = momentum_beta
        self.constraint_type = constraint_type
        self.enable_adaptive_tolerance = enable_adaptive_tolerance
        self.adaptive_tolerance_factor = adaptive_tolerance_factor
        
        # Lagrange multiplier
        self.lagrange_multiplier = lambda_init
        
        # EMA for constraint violation
        self.ema_constraint_violation = 0.0
        
        # Momentum for lambda updates
        self.lambda_momentum = 0.0
        
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
            "ema_violation": [],
        }
        
        # Rolling windows for detailed metrics
        self.length_history = []
        self.penalty_history = []
        self.violation_history = []
        self.lambda_history = []
        
        # Dynamic window size: precisely 2 batches if batch info provided
        if batch_size is not None and n_responses_per_prompt is not None:
            self.window_size = 2 * batch_size * n_responses_per_prompt
            print(f"\033[92mConstrainedDAPORewardManager: Setting window_size to {self.window_size} "
                  f"(2 batches × {batch_size} prompts × {n_responses_per_prompt} responses)\033[0m")
        else:
            self.window_size = 64  # Default fallback: approximately 2 batches for typical configs
            print(f"\033[93mConstrainedDAPORewardManager: Using default window_size={self.window_size}. "
                  f"Pass batch_size and n_responses_per_prompt for precise 2-batch window.\033[0m")
        
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
        - EMA update: $\tilde{g} = \alpha \cdot \tilde{g} + (1-\alpha) \cdot g$
        - Momentum update: $m = \beta \cdot m + (1-\beta) \cdot \tilde{g}$
        - Lambda update: $\lambda = \text{clip}(\lambda + \eta_\lambda \cdot m, \lambda_{min}, \lambda_{max})$
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
            
            if violation_ratio > self.tolerance:
                num_violations += 1
            
            # Apply penalty at the last token of the response
            # Formula: $\tilde{r}_T = r_T - \lambda \cdot (\frac{|y|}{L_{target}} - 1)$
            if valid_response_length > 0:
                penalty = self.lagrange_multiplier * violation_ratio
                reward_tensor[i, valid_response_length - 1] -= penalty
                total_penalty += abs(penalty)
                
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
            
        # Update EMA of constraint violation
        # Formula: $\tilde{g}^{(t+1)} = \alpha \cdot \tilde{g}^{(t)} + (1-\alpha) \cdot g^{(t)}$
        self.ema_constraint_violation = (
            self.ema_alpha * self.ema_constraint_violation + (1 - self.ema_alpha) * current_constraint_violation
        )
        
        # Update momentum for lambda
        # Formula: $m^{(t+1)} = \beta \cdot m^{(t)} + (1-\beta) \cdot \tilde{g}^{(t+1)}$
        self.lambda_momentum = (
            self.momentum_beta * self.lambda_momentum + (1 - self.momentum_beta) * self.ema_constraint_violation
        )
        
        # Update Lagrange multiplier with momentum
        # Formula: $\lambda^{(t+1)} = \text{clip}(\lambda^{(t)} + \eta_\lambda \cdot m^{(t+1)}, \lambda_{min}, \lambda_{max})$
        # Now using ratio-based tolerance: violations are relative to target
        if self.ema_constraint_violation > self.tolerance:
            # Increase lambda when constraint is violated (length ratio exceeds tolerance)
            self.lagrange_multiplier += self.lambda_lr * self.lambda_momentum
        elif self.ema_constraint_violation < -self.tolerance:
            # Decrease lambda when we're too far below target (ratio too small)
            self.lagrange_multiplier += self.lambda_lr * self.lambda_momentum
            
        # Clip lambda to valid range
        self.lagrange_multiplier = np.clip(
            self.lagrange_multiplier, self.lambda_min, self.lambda_max
        )
        
        # Adaptive tolerance (optional) - now in ratio form
        if self.enable_adaptive_tolerance:
            # Adjust tolerance based on current performance (relative standard deviation)
            length_std_ratio = np.std(response_lengths) / self.target_length
            self.tolerance = max(
                self.adaptive_tolerance_factor,  # Minimum tolerance ratio
                2 * length_std_ratio  # Two times the relative standard deviation
            )
        
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
        
        # Satisfaction rate (now using ratio-based tolerance)
        within_target = sum(1 for l in response_lengths if abs(l / self.target_length - 1.0) <= self.tolerance)
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
        if len(self.lambda_history) > 1:
            self.stats["lambda_change_rate"].append(
                (self.lambda_history[-1] - self.lambda_history[-2]) / self.lambda_lr
            )
        else:
            self.stats["lambda_change_rate"].append(0.0)
            
        self.stats["ema_violation"].append(self.ema_constraint_violation)
        
        # Update rolling windows
        self.length_history.extend(response_lengths)
        self.penalty_history.extend(penalties)
        self.violation_history.extend(constraint_violations)
        self.lambda_history.append(self.lagrange_multiplier)
        
        # Keep only recent history
        for hist in [self.length_history, self.penalty_history, self.violation_history]:
            if len(hist) > self.window_size * 10:
                hist[:] = hist[-self.window_size * 10:]
                
        if len(self.lambda_history) > self.window_size:
            self.lambda_history = self.lambda_history[-self.window_size:]
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    "response_lengths": response_lengths,
                    "constraint_violations": constraint_violations,
                    "avg_length": avg_length,
                    "max_length": max_length,
                    "lagrange_multiplier": self.lagrange_multiplier,
                    "ema_constraint_violation": self.ema_constraint_violation,
                    "num_violations": num_violations,
                    "penalty_magnitude": total_penalty / len(response_lengths),
                }
            }
        else:
            return reward_tensor
            
    def get_metrics(self) -> Dict[str, float]:
        """Get current constraint optimization metrics."""
        metrics = {}
        
        # Use rolling window for recent metrics
        if self.length_history:
            recent_lengths = self.length_history[-self.window_size:]
            metrics["constraint/avg_length"] = np.mean(recent_lengths)
            metrics["constraint/std_length"] = np.std(recent_lengths)
            metrics["constraint/min_length"] = np.min(recent_lengths)
            metrics["constraint/max_length"] = np.max(recent_lengths)
            metrics["constraint/p95_length"] = np.percentile(recent_lengths, 95)
            
            # Constraint satisfaction metrics (ratio-based)
            within_target = sum(1 for l in recent_lengths if abs(l / self.target_length - 1.0) <= self.tolerance)
            metrics["constraint/satisfaction_rate"] = within_target / len(recent_lengths)
            
            # Distance from target (relative)
            metrics["constraint/avg_relative_distance"] = np.mean([abs(l / self.target_length - 1.0) for l in recent_lengths])
            
        # Penalty statistics
        if self.penalty_history:
            recent_penalties = self.penalty_history[-self.window_size:]
            metrics["constraint/avg_penalty"] = np.mean(np.abs(recent_penalties))
            metrics["constraint/max_penalty"] = np.max(np.abs(recent_penalties))
            
            # Penalty ratio (how many samples have penalties)
            non_zero_penalties = [p for p in recent_penalties if abs(p) > 1e-6]
            if recent_penalties:
                metrics["constraint/penalty_active_rate"] = len(non_zero_penalties) / len(recent_penalties)
                if non_zero_penalties:
                    metrics["constraint/avg_active_penalty"] = np.mean(np.abs(non_zero_penalties))
                    
        # Lambda tracking
        if self.lambda_history:
            metrics["constraint/lambda"] = self.lambda_history[-1]
            if len(self.lambda_history) > 1:
                # Lambda change rate (moving average over recent steps)
                recent_changes = [self.lambda_history[i] - self.lambda_history[i-1] 
                                for i in range(max(1, len(self.lambda_history)-10), len(self.lambda_history))]
                metrics["constraint/lambda_change_rate"] = np.mean(recent_changes) / self.lambda_lr if recent_changes else 0
                
        # Violation statistics (ratio-based)
        if self.violation_history:
            recent_violations = self.violation_history[-self.window_size:]
            metrics["constraint/avg_violation_ratio"] = np.mean(recent_violations)
            metrics["constraint/violation_rate"] = sum(1 for v in recent_violations if v > self.tolerance) / len(recent_violations)
            
        # Add current values
        metrics["constraint/ema_violation_ratio"] = self.ema_constraint_violation
        metrics["constraint/lambda_momentum"] = self.lambda_momentum
        metrics["constraint/tolerance_ratio"] = self.tolerance
        metrics["constraint/target_length"] = self.target_length
        
        return metrics
        
    def reset_stats(self):
        """Reset tracking statistics."""
        for key in self.stats:
            self.stats[key] = []
        self.length_history = []
        self.penalty_history = []
        self.violation_history = []
        self.lambda_history = []
        
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
            
        if "constraint/ema_violation_ratio" in metrics:
            summary_parts.append(f"EMA Violation Ratio: {metrics['constraint/ema_violation_ratio']:.3f}")
            
        return " | ".join(summary_parts)