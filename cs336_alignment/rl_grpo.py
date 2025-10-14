"""
Group Relative Policy Optimization Implement
"""
from dataclasses import dataclass
from typing import Callable, List, Literal
import torch
import torch.nn as nn
import wandb


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    """
    Compute rewards for each group of rollout responses, normalized by the group size.
    
    Args:
        reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against
        the ground truths, producing a dict with keys"reward", "format_reward", and "answer_reward".
    
        rollout_responses: list[str] Rollouts from the policy. The length of this list is
        rollout_batch_size = n_prompts_per_rollout_batch * group_size.
    
        repeated_ground_truths: list[str] The ground truths for the examples. The length of this
        list is rollout_batch_size, because the ground truth for each example is repeated
        group_size times.

        group_size: int Number of responses per question (group).

        advantage_eps: float Small constant to avoid division by zero in normalization.
    
        normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise
        subtract only the group mean.
    
    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]].
            advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout response.
            raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout response.
            metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    
    Test:
        implement [adapters.run_compute_group_normalized_rewards]
        uv run pytest -k test_compute_group_normalized_rewards
    """
    raise NotImplementedError


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either
    the raw reward or an already-normalized advantage.
    
    Args:
        raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar
        reward/advantage for each rollout response.
        
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for
        each token.

    Returns:
        torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss (to
        be aggregated across the batch and sequence dimensions in the training loop).
        
    Test:
        implement [adapters.run_compute_naive_policy_gradient_loss]
        uv run pytest -k test_compute_naive_policy_gradient_loss
    """
    raise NotImplementedError


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
        
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log
        probs from the policy being trained.
        
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs
        from the old policy.

        cliprange: float Clip parameter ϵ (e.g. 0.2).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss torch.Tensor of shape (batch_size, sequence_length), the per-token clipped loss.
            metadata dict containing whatever you want to log. We suggest logging whether each
            token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of
            the min was lower than the LHS.
    
    Test:
        implement [adapters.run_compute_grpo_clip_loss]
        uv run pytest -k test_compute_grpo_clip_loss
    """
    raise NotImplementedError


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.

    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
        policy being trained.
        
        loss_type One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        
        raw_rewards Required if loss_type == "no_baseline"; shape (batch_size, 1).
        
        advantages Required for"reinforce_with_baseline" and "grpo_clip"; shape
        (batch_size, 1).
        
        old_log_probs Required for"grpo_clip"; shape (batch_size, sequence_length).
        
        cliprange Required for"grpo_clip"; scalar ϵ used for clipping.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss (batch_size, sequence_length), per-token loss.
            metadata dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
            
    Test:
        implement [adapters.run_compute_policy_gradient_loss]
        uv run pytest -k test_compute_policy_gradient_loss
    """
    raise NotImplementedError


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None= None,
) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those elements where mask == 1.
    
    Args:
        tensor: torch.Tensor The data to be averaged.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
        dim: int | None Dimension over which to average. If None, compute the mean over all
        masked elements.
    
    Returns:
        torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.
        
    Test:
        implement [adapters.run_masked_mean]
        uv run pytest -k test_masked_mean
    """
    raise NotImplementedError


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.
    
    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
        policy being trained.

        response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
        prompt/padding.
        
        gradient_accumulation_steps Number of microbatches per optimizer step.
        
        loss_type One of "no_baseline", "reinforce_with_baseline", "grpo_clip".
    
        raw_rewards Needed when loss_type == "no_baseline"; shape (batch_size, 1).

        advantages Needed when loss_type != "no_baseline"; shape (batch_size, 1).
        
        old_log_probs Required for GRPO-Clip; shape (batch_size, sequence_length).
        
        cliprange Clip parameter ϵ for GRPO-Clip.
        
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
            this so we can log it.

            metadata Dict with metadata from the underlying loss call, and any other statistics you
            might want to log.
    
    Test:
        implement [adapters.run_grpo_microbatch_train_step]
        uv run pytest -k test_grpo_microbatch_train_step
    """
    raise NotImplementedError

@dataclass
class GRPOConfig:
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4 # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 1024
    epochs_per_rollout_batch: int = 1 # On-policy
    train_batch_size: int = 256 # On-policy
    gradient_accumulation_steps: int = 128 # microbatch size is 2, will fit on H100
    gpu_memory_utilization: float = 0.85
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    ] = "reinforce_with_baseline"
    use_std_normalization: bool = True
    optimizer = torch.optim.AdamW(
        policy.parameters(), # type: ignore
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )


def grpo_train_loop():
    raise NotImplementedError

def grpo_off_policy():
    raise NotImplementedError

def grpo_off_policy_clip_ablation():
    raise NotImplementedError

