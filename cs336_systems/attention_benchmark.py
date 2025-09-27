import timeit
import itertools
from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch.profiler import ProfilerActivity
from cs336_basics.modules import MultiHeadSelfAttention
from cs336_basics.modules import AdamW
from profiling_benchmark import *


def run_attention(
    num_warmups: int = 5,
    num_trials: int = 100,
    batch_size: int = 8,
    d_models: list = [16, 32, 64, 128],
    seq_lens: list = [256, 1024, 4096, 8192, 16384],
) -> Callable:
    results = []

    for d_model, seq_len in itertools.product(d_models, seq_lens):
        # make cartesian product
        print(f"d_model={d_model}, seq_len={seq_len}")
        
        # Drop MultiHead ==> num_heads=1
        attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=1,
            rope_exist=False,
            device=get_device(),
            dtype=torch.float16,
        )
        
        # Optional: Compiler
        attn = torch.compile(attn)
        
        x = torch.randn(batch_size, seq_len, d_model, device=get_device(), dtype=torch.float16, requires_grad=True)
    
        optimizer = AdamW(attn.parameters())
        
        def run():
            # Wram_up
            for _ in range(num_warmups):
                out = attn(x)
                loss = out.sum()
                loss.backward()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
            
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.max_memory_allocated(get_device())
            # Forward
            start = timeit.default_timer()
            for _ in range(num_trials):
                out = attn(x)
                torch.cuda.synchronize()
            forward_time = (timeit.default_timer() - start) / num_trials
            mid_mem = torch.cuda.max_memory_allocated(get_device())
            
            # Backward
            start = timeit.default_timer()
            for _ in range(num_trials):
                out = attn(x)
                loss = out.sum()
                loss.backward()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
            backward_time = (timeit.default_timer() - start) / num_trials
            end_mem = torch.cuda.max_memory_allocated(get_device())
            
            results.append({
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "forward_time": forward_time,
                    "backward_time": backward_time,
                    "mem_fwd": (mid_mem - start_mem) / 1024**2,
                    "mem_bwd": (end_mem - mid_mem) / 1024**2,
            })
            
            print(f"Forward: {forward_time*1e3:.3f} ms, Backward: {backward_time*1e3:.3f} ms")
            print(f"Memory (fwd): {results[-1]['mem_fwd']:.2f} MB, (bwd): {results[-1]['mem_bwd']:.2f} MB")
            
        run()

    return results


def main():
    run_attention()

if __name__ == "__main__":
    main()