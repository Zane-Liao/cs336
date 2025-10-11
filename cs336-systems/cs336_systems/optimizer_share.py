import torch
from typing import Iterable, List, Type, Any, Optional, Dict
from torch.optim import Optimizer
import torch.distributed as dist

__all__ = [
    "OptimizerStateShare",
]


class OptimizerStateShare(Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], optimizer_cls: Type[Optimizer], **kwargs):
        if not dist.is_initialized():
            self.local_optimizer = optimizer_cls(params, **kwargs)
            self.all_params = list(params)
            self.param_to_rank = {id(p): 0 for p in self.all_params}
            self.world_size = 1
            return

        self.world_size = dist.get_world_size()
        rank = dist.get_rank()
        self.all_params = list(params)

        self.param_to_rank = {id(p): i % self.world_size for i, p in enumerate(self.all_params)}

        local_params = [p for p in self.all_params if self.param_to_rank[id(p)] == rank]
        self.local_optimizer = optimizer_cls(local_params, **kwargs)

    def zero_grad(self, set_to_none: bool = False):
        for p in self.all_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_()
                    p.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if dist.is_initialized() and self.world_size > 1:
            for p in self.all_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(self.world_size)

        self.local_optimizer.step()

        if dist.is_initialized() and self.world_size > 1:
            for p in self.all_params:
                owner_rank = self.param_to_rank[id(p)]
                dist.broadcast(p.data, src=owner_rank)

        return loss

    def state_dict(self):
        return {
            "local_optimizer": self.local_optimizer.state_dict(),
            "param_to_rank": self.param_to_rank,
        }

    def load_state_dict(self, state_dict):
        self.local_optimizer.load_state_dict(state_dict["local_optimizer"])
        self.param_to_rank = state_dict["param_to_rank"]

    def __getattr__(self, name):
        return getattr(self.local_optimizer, name)