"""
Direct Preference Optimization Implement
"""
from dataclasses import dataclass
from typing import Callable, List, Literal
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase
import wandb

__all__= [
    "look_at_hh",
    "dpo_loss",
    "dpo_training",
]

def look_at_hh():
    raise NotImplementedError


def dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Test:
        Implement the adapter [adapters.per_instance_dpo]
        uv run pytest -k test_per_instance_dpo_loss
    """
    raise NotImplementedError


def dpo_training():
    raise NotImplementedError