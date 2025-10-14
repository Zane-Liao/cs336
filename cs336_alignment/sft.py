"""
Supervised Fine-tuning
"""
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer
):
    """
    prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for
    other tokens (prompt or padding).
    
    Args:
        prompt_strs: list[str] List of prompt strings.
        output_strs: list[str] List of output strings.
        tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of
        the tokenized prompt and output strings. Then the returned dictionary should have the

    following keys:
        input_ids torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
        the tokenized prompt and output strings, with the final token sliced off.
        labels torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
        shifted input ids, i.e., the input ids without the first token.
        response_mask torch.Tensor of shape (batch_size, max(prompt_and_output_lens) -
        1): a mask on the response tokens in the labels.
    
    Test:
        implement [adapters.run_tokenize_prompt_and_output]
        uv run pytest -k test_tokenize_prompt_and_output
    """
    raise NotImplementedError


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    
    Args:
        logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
        containing unnormalized logits.

    Returns:
        torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
        prediction.
    
    Test:
        implement [adapters.run_compute_entropy]
        uv run pytest -k test_compute_entropy
    """
    raise NotImplementedError


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Args:
        model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device
        and in inference mode if gradients should not be computed).

        input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt +
        response tokens as produced by your tokenization method.

        labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your
        tokenization method.

        return_token_entropy: bool If True, also return per-token entropy by calling
        compute_entropy.

    Returns:
        dict[str, torch.Tensor].
            "log_probs" shape (batch_size, sequence_length), conditional log-probabilities
            log pθ (xt |x<t).

            "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy
            for each position (present only if return_token_entropy=True).
    
    Test:
        implement [adapters.run_get_response_log_probs]
        uv run pytest -k test_get_response_log_probs
    """
    raise NotImplementedError


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None= None,
) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only those elements where mask == 1.

    Args:
        tensor: torch.Tensor The tensor to sum and normalize.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
        normalize_constant: float the constant to divide by for normalization.
        dim: int | None the dimension to sum along before normalization. If None, sum over all
        dimensions.

    Returns:
        torch.Tensor the normalized sum, where masked elements (mask == 0) don’t contribute to
        the sum.
        
    Test:
        implement [adapters.run_masked_normalize]
        uv run pytest -k test_masked_normalize
    """
    raise NotImplementedError


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.

    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
        SFT policy being trained.
        response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
        prompt/padding.
        gradient_accumulation_steps Number of microbatches per optimizer step.
        normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
        this so we can log it.
        metadata Dict with metadata from the underlying loss call, and any other statistics you
        might want to log.
        
    Test:
        implement [adapters.run_sft_microbatch_train_step]
        uv run pytest -k test_sft_microbatch_train_step
    """
    raise NotImplementedError


def log_generations():
    raise NotImplementedError