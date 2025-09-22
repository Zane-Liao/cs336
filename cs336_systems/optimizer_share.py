import torch
import time
import os
from typing import List, Callable, Type, Any
from torch.optim import Optimizer
from torch.optim import Optimizer
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.fsdp

__all__ = [
    "optimizer_state_sharding",
    "OptimizerStateShare",
]


class OptimizerStateShare(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        raise NotImplementedError
    
    def step(self, closure, **kwargs):
        raise NotImplementedError
    
    def add_param_group(self, param_group: dict[str, Any]):
        raise NotImplementedError


def optimizer_state_sharding():
    raise NotImplementedError