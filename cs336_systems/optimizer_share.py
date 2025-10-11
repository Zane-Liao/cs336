import torch
from typing import List, Type, Any, Optional, Dict
from torch.optim import Optimizer
import torch.distributed as dist

__all__ = [
    "OptimizerStateShare",
]


class OptimizerStateShare(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        defaults = kwargs.pop("defaults", {})
        
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available")
        
        self._is_distributed = dist.is_initialized()
        if self._is_distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        
        self._optimizer_cls = optimizer_cls
        self._optimizer_kwargs = kwargs
        
        original_param_groups = self._normalize_param_groups(params)
        
        self._all_param_order: List[torch.nn.Parameter] = []
        seen_param_ids = set()
        for group in original_param_groups:
            for p in group['params']:
                if id(p) not in seen_param_ids:
                    self._all_param_order.append(p)
                    seen_param_ids.add(id(p))
        
        self._all_params = self._all_param_order
        
        self._owner_for: Dict[int, int] = {
            id(p): i % self.world_size
            for i, p in enumerate(self._all_param_order)
        }

        local_param_groups = []
        seen_local_param_ids = set()
        
        for group in original_param_groups:
            local_params_for_group = []
            for p in group['params']:
                if self._owner_for.get(id(p)) == self.rank and id(p) not in seen_local_param_ids:
                    local_params_for_group.append(p)
                    seen_local_param_ids.add(id(p))
            
            if local_params_for_group:
                local_group = {k: v for k, v in group.items() if k != 'params'}
                local_group['params'] = local_params_for_group
                local_param_groups.append(local_group)

        super().__init__(local_param_groups, defaults)

        local_opt_param_groups = []
        for group in self.param_groups:
            local_opt_group = {k: v for k, v in group.items() if k != 'params'}
            local_opt_group['params'] = list(group['params'])
            local_opt_param_groups.append(local_opt_group)
        
        self._local_opt = optimizer_cls(local_opt_param_groups, **self._optimizer_kwargs)
        
        self._use_ddp = False
    
    def _normalize_param_groups(self, params) -> List[Dict[str, Any]]:
        param_groups = list(params)
        if len(param_groups) == 0:
            return []
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]
        for group in param_groups:
            group["params"] = list(group["params"])
        return param_groups
    
    def zero_grad(self, set_to_none: bool = False):
        self._local_opt.zero_grad(set_to_none=set_to_none)
    
    def step(self, closure: Optional[Any] = None, **kwargs) -> Optional[float]:
        if not self._use_ddp and self.world_size > 1:
            for p in self._all_param_order:
                if p.grad is not None:
                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                    p.grad.data.div_(self.world_size)
        
        loss = self._local_opt.step(closure=closure, **kwargs)
        
        if self._is_distributed and self.world_size > 1:
            for p in self._all_param_order:
                owner = self._owner_for[id(p)]
                dist.broadcast(p.data, src=owner)
        
        return loss
    
    def add_param_group(self, param_group: Dict[str, Any]):
        pg_params = param_group['params']
        start_idx = len(self._all_param_order)
        
        new_unique_params = []
        current_param_ids = {id(p) for p in self._all_param_order}
        
        for p in pg_params:
            if id(p) not in current_param_ids:
                new_unique_params.append(p)
                current_param_ids.add(id(p))
        
        for i, p in enumerate(new_unique_params):
            owner_rank = (start_idx + i) % self.world_size
            self._owner_for[id(p)] = owner_rank
        
        self._all_param_order.extend(new_unique_params)
        
        local_params_for_new_group = [
            p for p in pg_params 
            if self._owner_for.get(id(p)) == self.rank
        ]
        
        if local_params_for_new_group:
            new_local_group = {k: v for k, v in param_group.items() if k != 'params'}
            new_local_group['params'] = local_params_for_new_group
            
            super().add_param_group(new_local_group)
            self._local_opt.add_param_group(new_local_group)
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            "local_opt_state": self._local_opt.state_dict(),
            "optimizer_cls": self._optimizer_cls,
            "optimizer_kwargs": self._optimizer_kwargs,
            "owner_for": self._owner_for,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._local_opt.load_state_dict(state_dict["local_opt_state"])
    
    def set_use_ddp(self, val: bool):
        self._use_ddp = bool(val)