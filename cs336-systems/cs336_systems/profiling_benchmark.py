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

__all__ = [
    "benchmark",
    "benchmarking",
    "benchmarking_mixed_precision",
    "memory_profiling",
    "mixed_precision_accumulation",
    "nsys_profile",
]


def benchmarking():
    raise NotImplementedError


def benchmark():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()
    timeit.default_timer()
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