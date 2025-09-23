import os
import time
import timeit
from typing import Callable
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity
from torch.utils.cpp_extension import load_inline
from cs336_basics.modules import TransformerLM
import torch.cuda.nvtx as nvtx
from dataclasses import dataclass

__all__ = [
    "benchmarking",
    "benchmarking_mixed_precision",
    "memory_profiling",
    "mixed_precision_accumulation",
    "nsys_profile",
]


@dataclass
class TransformerConfig:
    vocab_size: int = 10000
    context_length: int = 2048
    num_layers: int = 12
    d_model: int = 768
    num_heads: int = 12
    d_ff: int = 3072
    rope_theta: float = 10000.0


@dataclass
class BenchmarkConfig:
    batch_size: int = 4
    seq_len: int = 16
    num_steps: int = 10


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
        for step in range(num_steps):
            # Forward
            y = model(x).mean()

            # Backward
            model.zero_grad(set_to_none=True)
            y.backward()
            
            # asyn
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    return run


def benchmark(description: str, run: Callable, num_warmups: int = 10, num_trials: int = 3):
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
    return mean_time


def benchmarking():
    raise NotImplementedError


def nsys_profile():
    raise NotImplementedError


def mixed_precision_accumulation():
    raise NotImplementedError


def benchmarking_mixed_precision():
    raise NotImplementedError


def memory_profiling():
    raise NotImplementedError


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


if __name__ == "__main__":
    model_config = TransformerConfig()
    model = TransformerLM(**model_config.__dict__)
    mark_config = BenchmarkConfig()
    run_fn = run_transformer(model, **mark_config.__dict__)
    avg_time = benchmark("s", run_fn)
    print(f"Average step time: {avg_time:.3f} ms")