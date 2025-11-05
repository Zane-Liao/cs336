"""
Supervised Fine-tuning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

__all__ = [
    "tokenize_prompt_and_output",
    "compute_entropy",
    "get_response_log_probs",
    "masked_normalize",
    "sft_microbatch_train_step",
    "log_generations",
]

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
    """
    prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for
    other tokens (prompt or padding).
    
    Args:
        prompt_strs: list[str] List of prompt strings.
        output_strs: list[str] List of output strings.
        tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of
        the tokenized prompt and output strings. Then the returned dictionary should have the following keys:

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
    assert len(prompt_strs) == len(output_strs), "Must same!!!"
    
    batch_input_ids = [] 
    batch_response_masks = []
    
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        output_tokens = tokenizer(output, add_special_tokens=False)["input_ids"]
        
        full_tokens = prompt_tokens + output_tokens
        
        response_mask = [0] * len(prompt_tokens) + [1] * len(output_tokens)
        
        batch_input_ids.append(full_tokens)
        batch_response_masks.append(response_mask)
        
    max_len = max(len(x) for x in batch_input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    input_ids_paded = []
    response_mask_paded = []
    for input_ids, mask in zip(batch_input_ids, batch_response_masks):
        pad_len = max_len - len(input_ids)
        input_ids_paded.append(input_ids + [pad_id] * pad_len)
        response_mask_paded.append(mask + [0] * pad_len)
    
    input_ids = torch.tensor(input_ids_paded, dtype=torch.float32)
    response_mask = torch.tensor(response_mask_paded, dtype=torch.float32)
    
    labels = input_ids.clone()
    input_ids = input_ids[:, :-1]
    labels = labels[:, 1:]
    response_mask = response_mask[:, 1:]
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
        }


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
    probs = F.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs), dim=-1)


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
    # (batch, seq_len) -> (batch, seq_len, 1) -> gather(unsqueeze) -> (batch, seq_len, 1) -> squeeze -> (batch, seq_len)
    outputs = model(input_ids)
    logits = outputs.logits
    
    log_probs = F.log_softmax(logits, dim=-1)
    
    token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    
    result = {"log_probs": token_log_probs}

    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
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
    t_masked = tensor * mask
    
    summed = t_masked.sum() if dim is None else t_masked.sum(dim=dim)
    
    return summed / normalize_constant


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
        **loss** scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
        this so we can log it.
        **metadata** Dict with metadata from the underlying loss call, and any other statistics you
        might want to log.
        
    Test:
        implement [adapters.run_sft_microbatch_train_step]
        uv run pytest -k test_sft_microbatch_train_step
    """
    loss = masked_normalize(
        tensor=-policy_log_probs,
        mask=response_mask,
        dim=None,
        normalize_constant=normalize_constant
    )
    
    batch_size = policy_log_probs.shape[0]
    
    sft_avg_loss = loss / batch_size if batch_size > 0 else loss
    
    scale_loss = sft_avg_loss / gradient_accumulation_steps
    
    scale_loss.backward()
    
    metadata = {
        "sft_loss" : scale_loss.detach()
    }
    
    return (scale_loss.detach(), metadata)


def log_generations():
    raise NotImplementedError