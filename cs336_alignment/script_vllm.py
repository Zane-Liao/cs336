"""
vLLM script
"""
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

from sft_method import *

__all__ = [
    "evaluate_vllm",
    "init_vllm",
    "load_policy_into_vllm_instance",
]

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list,
    eval_sampling_params: SamplingParams,
    references: list = None,
    output_file: str = "eval_math_result.json"
):
    """
    Evaluates a VLLM model on a given dataset and reward function.

    Args:
        vllm_model: The VLLM model instance.
        reward_fn: A function that takes a prediction and a reference string and returns a dictionary of metrics.
        prompts: A list of prompt strings.
        eval_sampling_params: Sampling parameters for VLLM generation.
        references: A list of reference strings corresponding to the prompts.
        output_file: The path to save the JSON evaluation results.
    """
    import json
    results = []

    # Limit evaluation to the first 10 samples for a quick test run
    # You can remove this line to evaluate the entire dataset
    # Sample ==> 590705
    prompts = prompts[:10]
    if references:
        references = references[:10]

    print(f"Starting evaluation for {len(prompts)} samples...")

    for i, prompt in enumerate(prompts):
        formatted_prompt = f"q: {prompt}" # Using a simpler prompt format

        # Step 3: Generate output from the model
        outputs = vllm_model.generate([formatted_prompt], sampling_params=eval_sampling_params)
        
        # The output from vllm.generate is a list of RequestOutput objects.
        # Each RequestOutput has a list of CompletionOutput objects in its `outputs` attribute.
        # We need to access the text from the first CompletionOutput.
        generated_text = outputs[0].outputs[0].text.strip()

        # Step 4: Calculate metrics
        if references is not None:
            metrics = reward_fn(generated_text, references[i])
        else:
            # If no references, just record the length as a basic metric
            metrics = {"generated_length": len(generated_text.split())}
        
        print(f"Processed sample {i+1}/{len(prompts)}")

        results.append({
            "prompt": prompt,
            "formatted_prompt": formatted_prompt,
            "generated": generated_text,
            "reference": references[i] if references else "N/A",
            "metrics": metrics
        })

    # Step 5: Save results to a file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Evaluation finished. Results saved to {output_file}")


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
    

# # Setup wandb metrics
# wandb.define_metric("train_step")
# wandb.define_metric("eval_step") # the x‑axis for training
# # the x‑axis for evaluation
# # everything that starts with train/ is tied to train_step
# wandb.define_metric("train/*", step_metric="train_step")
# # everything that starts with eval/ is tied to eval_step
# wandb.define_metric("eval/*", step_metric="eval_step")


def sft():
    raise NotImplementedError

def expert_iteration():
    # sampling_min_tokens = 4
    # sampling_params = SamplingParams(
    #     temperature=sampling_temperature,
    #     max_tokens=sampling_max_tokens,
    #     min_tokens=sampling_min_tokens,
    #     n=G,
    #     seed=seed,
    # )
    raise NotImplementedError

# --------------------------------------------------------------------------------------------

def evaluate_vllm_main():
    # 1. Load the dataset
    ds = load_dataset("hkust-nlp/dart-math-uniform")
    print("Dataset loaded.")
    print("Example data point:", ds["train"][0])

    # 2. Initialize the VLLM model
    # Make sure you have enough VRAM for this model
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