import os
import sys
import math
import time
import numpy as np
from inspect import isfunction
from typing import List, Callable, Set
import torch
import torch.nn as nn
import torch.distributed.fsdp
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import ProfilerActivity
from torch.utils.cpp_extension import load_inline
from torch.nn.parallel import DistributedDataParallel

__all__ = [
    "DDPIndividualParameters",
    "BucketDDPIndividualParameters",
]


class DDPIndividualParameters(nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Initial DDPIndividualParameters !!!")
        
        self.module = module
        
        self.rank = dist.get_rank()
        
        self.world_size = dist.get_world_size()
        
        self._params: List[nn.Parameter] = [p for _, p in module.named_parameters()]
        
        self._buffers = [b for _, b in module.named_buffers()]
        
        self._handles: List[dist.Work] = []
        
        self._hook_handles = []
                
        self._boardcast_parameters_and_buffers()

        self._register_hooks()
        
    def _boardcast_parameters_and_buffers(self):
        seen_params: Set[int] = set()
        for p in self._params:
            if id(p) in seen_params:
                continue
            dist.broadcast(p.data, src=0)
            seen_params.add(id(p))
        
        for b in self._buffers:
            try:
                dist.broadcast(b.data, src=0)
            except Exception:
                pass
    
    def _register_hooks(self):
        seen_params: Set[int] = set()
        for p in self._params or id(p) in seen_params:
            if not p.requires_grad:
                continue
            seen_params.add(id(p))
            
            def make_hook(param):
                def hook(grad):
                    if grad is None:
                        return None
                    
                    handle = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
                    
                    self._handles.append(handle)
                    
                    return grad
                return hook
            
            h = p.register_hook(make_hook(p))
            self._hook_handles.append(h)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for h in self._handles:
            h.wait()
        
        self._handles = []
        
        if self.world_size > 1:
            for p in self._params:
                if not p.requires_grad:
                    if p.grad is not None:
                        p.grad.div_(self.world_size)


class BucketDDPIndividualParameters(nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
    
    def forward(self, *inputs, **kwargs):
        raise NotImplementedError
    
    def finish_gradient_synchronization(self):
        raise NotImplementedError


def get_device(index: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")