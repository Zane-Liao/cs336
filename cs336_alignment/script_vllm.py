"""
vLLM SFT script
"""
from dataclasses import dataclass
import os
import json
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
from unittest.mock import patch
from collections import Counter
from datasets import load_dataset
from typing import Any, Callable, Dict, Callable, Literal, Optional
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer

from sft_method import *
from rl_grpo import *

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

# --------------------------------------------------------------------------------------------

def sft():
    model_name = "Qwen/Qwen2.5-Math-1.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_dataset("hkust-nlp/dart-math-uniform")

    num_samples = 200
    ds_sample = ds["train"].shuffle(seed=42).select(range(num_samples))

    # openai/gsm8k -> "question" "answer"
    # 
    prompts = ds_sample["query"]
    responses = ds_sample["response"]
    
    generation_kwargs = {
        "max_new_tokens": 256,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    results = log_generations(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        ground_truth_responses=responses,
        reward_fn=simple_reward_fn,
        device=device,
        generation_kwargs=generation_kwargs,
    )
    
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/sft_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("SFT logging complete. Results saved to outputs/sft_result.json")

# --------------------------------------------------------------------------------------------

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

def evaluate_vllm():
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

def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    ground_truth_responses: list[str],
    reward_fn: Callable,
    device: torch.device,
    generation_kwargs: Optional[dict] = None,
    batch_size: int = 8
):
    if generation_kwargs is None:
        generation_kwargs = {
            "max_new_tokens": 256, 
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    
    model.eval()
    all_logs = []
    
    total_response_len = 0
    total_correct_len = 0
    total_incorrect_len = 0
    total_correct_count = 0
    total_incorrect_count = 0
    total_reward = 0.0
    total_entropy = 0.0

    original_padding_side = tokenizer.padding_side
    
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Logging Generations"):
            batch_prompts = prompts[i : i + batch_size]
            batch_ground_truth = ground_truth_responses[i : i + batch_size]

            tokenizer.padding_side = "left"
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(device)

            inputs['input_ids'] = inputs['input_ids'].long()
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'].long()

            input_len = inputs.input_ids.shape[1]

            outputs = model.generate(**inputs, **generation_kwargs)

            generated_ids_batch = outputs[:, input_len:]
            
            generated_responses = tokenizer.batch_decode(
                generated_ids_batch, skip_special_tokens=True
            )

            batch_rewards = []
            batch_is_correct = []
            for j in range(len(batch_prompts)):
                reward_info = reward_fn(
                    batch_prompts[j], 
                    generated_responses[j], 
                    batch_ground_truth[j]
                )
                batch_rewards.append(reward_info)
                
                is_correct = reward_info.get("answer_reward", 0.0) == 1.0
                batch_is_correct.append(is_correct)
                
                total_reward += reward_info.get("total_reward", 0.0)

            tokenizer.padding_side = "right"

            tokenized = tokenize_prompt_and_output(
                batch_prompts, generated_responses, tokenizer
            )
            
            t_input_ids = tokenized["input_ids"].long().to(device)
            t_labels = tokenized["labels"].long().to(device)
            t_response_mask = tokenized["response_mask"].long().to(device)

            log_probs_data = get_response_log_probs(
                model, t_input_ids, t_labels, return_token_entropy=True
            )
            
            token_entropy_batch = log_probs_data["token_entropy"] # [batch_size, seq_len]

            token_entropy_batch = torch.nan_to_num(token_entropy_batch, nan=0.0)
            
            num_response_tokens_batch = t_response_mask.sum(dim=1) # [batch_size,]
            sum_entropy_batch = (token_entropy_batch * t_response_mask).sum(dim=1) # [batch_size,]
            
            safe_num_tokens = num_response_tokens_batch.clamp(min=1.0)
            avg_entropy_batch = sum_entropy_batch / safe_num_tokens
            avg_entropy_batch[num_response_tokens_batch == 0] = 0.0
            
            avg_entropy_list = avg_entropy_batch.cpu().tolist()
            total_entropy += avg_entropy_batch.sum().item()
            
            response_lens_tensor = (generated_ids_batch != tokenizer.pad_token_id).sum(dim=1)
            response_lens = response_lens_tensor.cpu().tolist()
            
            total_response_len += response_lens_tensor.sum().item()

            for j in range(len(batch_prompts)):
                log_entry = {
                    "prompt": batch_prompts[j],
                    "ground_truth_response": batch_ground_truth[j],
                    "generated_response": generated_responses[j],
                    "reward_info": batch_rewards[j],
                    "avg_token_entropy": avg_entropy_list[j],
                }
                all_logs.append(log_entry)
                
                response_len = response_lens[j]
                if batch_is_correct[j]:
                    total_correct_len += response_len
                    total_correct_count += 1
                else:
                    total_incorrect_len += response_len
                    total_incorrect_count += 1
    
    tokenizer.padding_side = original_padding_side
    
    num_examples = len(prompts)
    aggregate_stats = {
        "avg_response_length": (total_response_len / num_examples) if num_examples > 0 else 0,
        "avg_correct_length": (total_correct_len / total_correct_count) if total_correct_count > 0 else 0,
        "avg_incorrect_length": (total_incorrect_len / total_incorrect_count) if total_incorrect_count > 0 else 0,
        "avg_reward": (total_reward / num_examples) if num_examples > 0 else 0,
        "avg_entropy": (total_entropy / num_examples) if num_examples > 0 else 0,
        "accuracy": (total_correct_count / num_examples) if num_examples > 0 else 0,
    }

    return {
        "per_example_logs": all_logs,
        "aggregate_stats": aggregate_stats
    }


def simple_reward_fn(prompt: str, generated_response: str, ground_truth: str) -> Dict[str, Any]:
    gen = generated_response.strip().lower()
    gt = ground_truth.strip().lower()

    if gen.startswith("answer:"):
        format_reward = 1.0
    else:
        format_reward = 0.5

    gen_tokens = gen.split()
    gt_tokens = gt.split()
    if len(gt_tokens) == 0:
        answer_reward = 0.0
    else:
        common_tokens = sum((Counter(gt_tokens) & Counter(gen_tokens)).values())
        answer_reward = common_tokens / max(len(gt_tokens), 1)

    total_reward = format_reward + answer_reward

    return {
        "format_reward": format_reward,
        "answer_reward": answer_reward,
        "total_reward": total_reward,
    }
    
# --------------------------------------------------------------------------------------------

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


def grpo_train_loop(
    n_grpo_steps: int | None = None,
    learning_rate: float | None = None,
    advantage_eps: float | None = None,
    rollout_batch_size: int | None = None,
    group_size: int | None = None,
    sampling_temperature: float | None = None,
    sampling_min_tokens: int | None = None,
    sampling_max_tokens: int | None = None,
    epochs_per_rollout_batch: int | None = None,
    train_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    gpu_memory_utilization: float | None = None,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] | None = None,
    use_std_normalization: bool | None = None,
    optimizer: torch.optim.AdamW | None = None,
):
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    raise NotImplementedError


def grpo_off_policy():
    raise NotImplementedError

def grpo_off_policy_clip_ablation():
    raise NotImplementedError


def grpo(grpo_type: str | None = None):
    if grpo_type == "on-policy":
        raise NotImplementedError
    elif grpo_type == "off-policy":
        raise NotImplementedError
    else:
        raise ValueError("!!!")


# # Setup wandb metrics
# wandb.define_metric("train_step")
# wandb.define_metric("eval_step") # the x‑axis for training
# # the x‑axis for evaluation
# # everything that starts with train/ is tied to train_step
# wandb.define_metric("train/*", step_metric="train_step")
# # everything that starts with eval/ is tied to eval_step
# wandb.define_metric("eval/*", step_metric="eval_step")


if __name__ == "__main__":
    # evaluate_vllm()
    # sft()
    # expert_iteration()
    grpo(grpo_type="off-policy")