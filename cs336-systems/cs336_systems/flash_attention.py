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


class FlashAttnAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        raise NotImplementedError

    @staticmethod
    def backward(self):
        raise NotImplementedError


class TritonFlashAttentionAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(self):
        raise NotImplementedError
    
    @staticmethod
    def backward(self):
        raise NotImplementedError


# Pytorch Implement
def flash_attn_forward():
    raise NotImplementedError


# Pytorch Implement
def flash_attn_backward():
    raise NotImplementedError


# Triton Implement
@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    raise NotImplementedError


@triton.jit
def flash_bwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    DQ_ptr,
    DK_ptr,
    DV_ptr,
    DO_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    raise NotImplementedError