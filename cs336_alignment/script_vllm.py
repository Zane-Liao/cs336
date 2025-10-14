"""
vLLM script
"""
from typing import Callable, List
import torch
import torch.nn as nn
from unittest.mock import patch
from transformers import PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import wandb


def evaluate_vllm(
  vllm_model: LLM,
  reward_fn: Callable[[str, str], dict[str, float]],
  prompts: List[str],
  eval_sampling_params: SamplingParams
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    raise NotImplementedError
  

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )

    with world_size_patch, profiling_patch:
        return LLM(
        model=model_id,
        device=device,
        dtype=torch.bfloat16,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()

    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
    

# Setup wandb metrics
wandb.define_metric("train_step")
wandb.define_metric("eval_step") # the x‑axis for training
# the x‑axis for evaluation
# everything that starts with train/ is tied to train_step
wandb.define_metric("train/*", step_metric="train_step")
# everything that starts with eval/ is tied to eval_step
wandb.define_metric("eval/*", step_metric="eval_step")

