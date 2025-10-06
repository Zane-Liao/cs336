import os
import time
import numpy as np
from typing import Callable
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity
from torch.utils.cpp_extension import load_inline
import torch.distributed as dist
import torch.distributed.fsdp
from torch.nn.parallel import DistributedDataParallel

__all__ = [
    "DDPIndividualParameters",
    "BucketDDPIndividualParameters",
]


class DDPIndividualParameters(nn.Module):
    def __init__(self, module: torch.nn.Module):
        raise NotImplementedError
    
    def forward(self, *inputs, **kwargs):
        raise NotImplementedError
    
    def finish_gradient_synchronization(self):
        raise NotImplementedError


class BucketDDPIndividualParameters(nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        raise NotImplementedError
    
    def forward(self, *inputs, **kwargs):
        raise NotImplementedError
    
    def finish_gradient_synchronization(self):
        raise NotImplementedError


def distributed_communication_single_node():
    raise NotImplementedError


def naive_ddp():
    raise NotImplementedError


def naive_ddp_benchmarking():
    raise NotImplementedError


def minimal_ddp_flat_benchmarking():
    raise NotImplementedError


def ddp_overlap_individual_parameters():
    raise NotImplementedError


def ddp_overlap_individual_parameters_benchmarking():
    raise NotImplementedError


def ddp_overlap_bucketed():
    raise NotImplementedError


def ddp_bucketed_benchmarking():
    raise NotImplementedError


def communication_accounting():
    raise NotImplementedError
