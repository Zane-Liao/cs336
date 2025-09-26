import os
import time
import math
import timeit
from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch.profiler import ProfilerActivity
from torch.utils.cpp_extension import load_inline
from cs336_basics.modules import TransformerLM
from cs336_basics.modules import AdamW
import torch.cuda.nvtx as nvtx
from dataclasses import dataclass
import torch.utils.checkpoint as cp

__all__ = [
    "benchmark",
    "profile",
    "run_transformer",
    "run_transformer_nvtx",
    "profile_memory"
]


@dataclass
class TransformerConfig:
    vocab_size: int = 10000
    context_length: int = 128
    num_layers: int = 24
    d_model: int = 1024
    num_heads: int = 16
    d_ff: int = 4096
    rope_theta: float = 10000.0


@dataclass
class BenchmarkConfig:
    batch_size: int = 4
    seq_len: int = 16
    num_steps: int = 10
    
    
@dataclass
class NvtxConfig:
    batch_size: int = 4
    seq_len: int = 4
    num_steps: int = 5
    use_optimizer: bool = True


def run_transformer(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    num_steps: int,
    ) -> Callable:
    model = model.to(get_device())
    model.train()
    
    vocab_size = model.vocab_size
    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=get_device()
    )
    
    def run():
        forward_times, backward_times = [], []
        for step in range(num_steps):
            # Forward
            start = timeit.default_timer()
            y = model(x).mean()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_times.append((timeit.default_timer() - start) * 1000)

            # Backward
            model.zero_grad(set_to_none=True)
            start = timeit.default_timer()
            y.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_times.append((timeit.default_timer() - start) * 1000)
            count = step
            print(f"{count} Forward avg: {mean(forward_times):.3f} ms, Forward sd: {sd(forward_times):.3f} ms")
            print(f"{count} Backward avg: {mean(backward_times):.3f} ms, Backward sd: {sd(backward_times):.3f} ms")

    return run


def run_transformer_nvtx(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    num_steps: int,
    use_optimizer: bool = False,
    train: bool = True,
    ) -> Callable:
    model = model.to(get_device())
    model.train()
    
    vocab_size = model.vocab_size
    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=get_device()
    )
    
    # Initialize optimizer if requested
    optimizer = AdamW(model.parameters()) if use_optimizer else None
    scaler = torch.amp.GradScaler("cuda") if use_optimizer else None
    
    def run():
        for step in range(num_steps):
            nvtx.range_push(f"step_{step}")

            # Forward
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with nvtx.range("forward"):
                    y = cp.checkpoint(model, x)  # Optional
                    y = model(x).mean()

            # Backward
            if train:
                if optimizer:
                    optimizer.zero_grad()
                    with nvtx.range("backward"):
                        scaler.scale(y).backward()
                    with nvtx.range("optimizer_step"):
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    model.zero_grad(set_to_none=True)
                    with nvtx.range("backward"):
                        y.backward()
                        
            torch.cuda.empty_cache()
            nvtx.range_pop()

    return run

def profile_memory(run_fn, snapshot_path="memory_snapshot.pickle", max_entries=1000_000):
    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    print("[Memory Profiler] Recording started...")
    
    run_fn()
    
    torch.cuda.memory._dump_snapshot(snapshot_path)
    print(f"[Memory Profiler] Snapshot saved to {snapshot_path}")
    
    torch.cuda.memory._record_memory_history(enabled=None)
    print("[Memory Profiler] Recording stopped.")
    
    print("\n[Memory Summary Report]")
    print(torch.cuda.memory_summary(device=torch.cuda.current_device()))


def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
    for _ in range(num_warmups):
        run()
        
    times: list[float] = []
    for trail in range(num_trials):
        start_time = timeit.default_timer()
        
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = timeit.default_timer()
        times.append((end_time - start_time) * 1000)
        
    mean_time = mean(times)
    sd_time = sd(times)
    return mean_time, sd_time

#############################################################################################################
#############################################################################################################

def get_device(index: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")
    
def run_operation1(dim: int, operation: Callable) -> Callable:
    x = torch.randn(dim, dim, device=get_device())
    return lambda : operation(x)

def run_operation2(dim: int, operation: Callable) -> Callable:
    x = torch.randn(dim, dim, device=get_device())
    y = torch.randn(dim, dim, device=get_device())
    return lambda : operation(x, y)

def mean(x: list[float]) -> float:
    return sum(x) / len(x)

def sd(data: list[float]) -> float:
    mean_val = sum(data) / len(data)
    variance = sum((x - mean_val)**2 for x in data) / len(data)
    return variance**0.5

def profile(description: str, run: Callable, num_warmups: int = 1, with_stack: bool = False):
    # Warmup
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
    # Run the code with the profiler
    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # Output stack trace for visualization
            with_stack=with_stack,
            # Needed to export stack trace for visualization
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
    # Print out table
    table = prof.key_averages().table(sort_by="cuda_time_total",
                                      max_name_column_width=80,
                                      row_limit=10)
    #text(f"## {description}")
    #text(table, verbatim=True)
    # Write stack trace visualization
    if with_stack:
        text_path = f"var/stacks_{description}.txt"
        svg_path = f"var/stacks_{description}.svg"
        prof.export_stacks(text_path, "self_cuda_time_total")
    return table

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    query: Tensor,
    key: Tensor, 
    value: Tensor, 
    mask: Tensor | None = None,
) -> Tensor:
    d_k = key.shape[-1]
    with nvtx.range("computing attention scores"):
        scores = torch.einsum("... q d, ... k d -> ... q k", query, key) / math.sqrt(d_k)
    
    if mask is not None:
        with nvtx.range("apply musk"):
            scores = scores.masked_fill(mask == 0, float('-inf'))

    with nvtx.range("computing softmax"):
        scores = torch.softmax(scores, dim=-1)
    
    with nvtx.range("final matmul"):
        output = scores @ value
    
    return output

def main():
    if torch.cuda.is_available():
        print("Running the GPU")
        model_config = TransformerConfig()
        model = TransformerLM(**model_config.__dict__)
        nvtx_config = NvtxConfig()
        run_fn_nvtx = run_transformer_nvtx(model, **nvtx_config.__dict__)
        profile_memory(run_fn_nvtx, "train_snapshot.pickle")
    else:
        print("Running the CPU")
        model_config = TransformerConfig()
        model = TransformerLM(**model_config.__dict__)
        mark_config = BenchmarkConfig()
        run_fn = run_transformer(model, **mark_config.__dict__)
        avg_time, sd_time = benchmark("s", run_fn)
        print(f"Average step time: {avg_time:.3f} ms, Sd step time: {sd_time:.3f} ms")

if __name__ == "__main__":
    main()