import math
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

__all__ = [
    "FlashAttnAutogradFunction",
    "TritonFlashAttentionAutogradFunction",
]


class FlashAttnAutogradFunction(nn.Module):
    def __init__(self):
        raise NotImplementedError
    
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class TritonFlashAttentionAutogradFunction(nn.Module):
    def __init__(self):
        raise NotImplementedError
    
    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError


# Pytorch Implement
def flash_forward():
    raise NotImplementedError


# Triton Implement
def triton_flash_forward():
    raise NotImplementedError


def causal_masking():
    raise NotImplementedError


# Pytorch Implement
def flash_backward():
    raise NotImplementedError


def flash_benchmarking():
    raise NotImplementedError