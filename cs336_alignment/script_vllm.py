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
        # --- FIX ENDS HERE ---

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

