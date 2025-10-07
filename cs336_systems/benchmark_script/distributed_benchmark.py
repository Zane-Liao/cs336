import os
import sys
import math
import time
import torch
from inspect import isfunction
from typing import List, Callable
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.fsdp
import torch.multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt 


def benchmarking_main():
    # "gloo" ==> cpu "nccl" ==> gpu
    # 1MB 10MB 100MB 1GB 
    data_sizes = [1 << 20, 10 << 20, 100 << 20, 1 << 30]
    world_size = [2, 4, 6]
    
    # results = []
    
    for size in data_sizes:
        num_elements = size  // 4
        for ws in world_size:
            print(f"\n--- Running | size={size/1e6:.1f}MB | world_size={ws} ---\n")
            spawn(all_reduce, ws, num_elements=num_elements)
            # results.append({
            #     "device": get_device(),
            #     "data_MB": size / (1 << 20),
            #     "world_size": ws,
            # })


# Percy
def all_reduce(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)
    # Create tensor
    tensor = torch.randn(num_elements, device=get_device(rank))
    # Warmup
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kernels to finish
        dist.barrier()            # Wait for all the processes to get here
    # Perform all-reduce
    start_time = time.time()
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kernels to finish
        dist.barrier()            # Wait for all the processes to get here
    end_time = time.time()
    duration = end_time - start_time
    print(f"[all_reduce] Rank {rank}: all_reduce(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)
    # Measure the effective bandwidth
    dist.barrier()
    size_bytes = tensor.element_size() * tensor.numel()
    sent_bytes = size_bytes * 2 * (world_size - 1)  # 2x because send input and receive output
    total_duration = world_size * duration
    bandwidth = sent_bytes / total_duration
    print(f"[all_reduce] Rank {rank}: all_reduce measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)
    cleanup()


# Percy
def reduce_scatter(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)
    # Create input and outputs
    input = torch.randn(world_size, num_elements, device=get_device(rank))  # Each rank has a matrix
    output = torch.empty(num_elements, device=get_device(rank))
    # Warmup
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kerels to finish
        dist.barrier()            # Wait for all the processes to get here
    # Perform reduce-scatter
    start_time = time.time()
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kerels to finish
        dist.barrier()            # Wait for all the processes to get here
    end_time = time.time()
    duration = end_time - start_time
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)
    # Measure the effective bandwidth
    dist.barrier()
    data_bytes = output.element_size() * output.numel()  # How much data in the output
    sent_bytes = data_bytes * (world_size - 1)  # How much needs to be sent (no 2x here)
    total_duration = world_size * duration  # Total time for transmission
    bandwidth = sent_bytes / total_duration
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)
    cleanup()


# Percy
def generate_sample_data():
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)
    return data


# Percy
def setup(rank: int, world_size: int):
    # Specify where master lives (rank 0), used to coordinate (actual data goes through NCCL)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"
    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        

def cleanup():
    torch.distributed.destroy_process_group()

def get_device(index: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")

class DisableDistributed:
    """Context manager that temporarily disables distributed functions (replaces with no-ops)"""
    def __enter__(self):
        self.old_functions = {}
        for name in dir(dist):
            value = getattr(dist, name, None)
            if isfunction(value):
                self.old_functions[name] = value
                setattr(dist, name, lambda *args, **kwargs: None)

    def __exit__(self, exc_type, exc_value, traceback):
        for name in self.old_functions:
            setattr(dist, name, self.old_functions[name])

def spawn(func: Callable, world_size: int, *args, **kwargs):
    # Note: assume kwargs are in the same order as what main needs
    if sys.gettrace():
        # If we're being traced, run the function directly, since we can't trace through mp.spawn
        with DisableDistributed():
            args = (0, world_size,) + args + tuple(kwargs.values())
            func(*args)
    else:
        args = (world_size,) + args + tuple(kwargs.values())
        mp.spawn(func, args=args, nprocs=world_size, join=True)

def int_divide(a: int, b: int):
    """Return a / b and throw an error if there's a remainder."""
    assert a % b == 0
    return a // b

def summarize_tensor(tensor: torch.Tensor) -> str:
    return "x".join(map(str, tensor.shape)) + "[" + str(round(tensor.view(-1)[0].item(), 4)) + "...]"

def get_init_params(num_inputs: int, num_outputs: int, rank: int) -> nn.Parameter:
    torch.random.manual_seed(0)  # For reproducibility
    return nn.Parameter(torch.randn(num_inputs, num_outputs, device=get_device(rank)) / math.sqrt(num_outputs))

def render_duration(duration: float) -> str:
    if duration < 1e-3:
        return f"{duration * 1e6:.2f}us"
    if duration < 1:
        return f"{duration * 1e3:.2f}ms"
    return f"{duration:.2f}s"

if __name__ == "__main__":
    benchmarking_main()