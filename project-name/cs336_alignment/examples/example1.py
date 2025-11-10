from datasets import load_dataset
import json
from typing import Callable, List
import torch
import torch.nn as nn
from unittest.mock import patch
from transformers import PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import wandb
from datasets import load_dataset
from script_vllm import evaluate_vllm

# 1. Load the dataset
ds = load_dataset("hkust-nlp/dart-math-uniform")
print("Dataset loaded.")
print("Example data point:", ds["train"][0])

# 2. Initialize the VLLM model
print("Loading VLLM model...")
llm_model = LLM("Qwen/Qwen2.5-Math-1.5B", tensor_parallel_size=1)
print("Model loaded.")

# 3. Define sampling parameters
sampling = SamplingParams(max_tokens=1024, temperature=1.0, top_p=1.0) # Set temperature to 0 for more deterministic output

# 4. Define the reward function for evaluation
def reward_fn(pred: str, ref: str) -> dict[str, float]:
    """Calculates exact match between prediction and reference."""
    return {"exact_match": float(pred.strip() == ref.strip())}

# 5. Prepare prompts and references from the dataset
prompts = [item["query"] for item in ds["train"]]
references = [item["response"] for item in ds["train"]]

# 6. Run the evaluation
evaluate_vllm(llm_model, reward_fn, prompts, sampling, references=references)