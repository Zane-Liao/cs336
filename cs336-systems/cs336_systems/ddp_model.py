from typing import List, Set, Dict, Any
import torch
import torch.nn as nn
import torch.distributed as dist

__all__ = [
    "DDPIndividualParameters",
    "BucketDDPIndividualParameters",
]


class DDPIndividualParameters(nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Not Initial DDPIndividualParameters !!!")
        
        self.module = module
        
        self.rank = dist.get_rank()
        
        self.world_size = dist.get_world_size()
        
        self._params: List[nn.Parameter] = [p for _, p in module.named_parameters()]
        
        self._buffers = [b for _, b in module.named_buffers()]

        self._hook_handles = []
                
        self._broadcast_parameters_and_buffers()

        self._register_hooks()
        
    def _broadcast_parameters_and_buffers(self):
        seen_params: Set[int] = set()
        with torch.no_grad():
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
        for p in self._params:
            if id(p) in seen_params or not p.requires_grad:
                continue
            seen_params.add(id(p))
            
            def make_hook(param):
                def hook(grad):
                    if grad is None:
                        return
                    
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                    
                    grad.div_(self.world_size)
                    
                    return grad
                return hook
            
            handle = p.register_hook(make_hook(p))
            self._hook_handles.append(handle)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        raise NotImplementedError


# ERROR🌚🌚🌚 BucketDDPIndividualParameters Impl
class BucketDDPIndividualParameters(nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float = 25.0):
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Not Initial BucketDDPIndividualParameters !!!")
        
        self.module = module
        
        self._bucket: Set = set()
        
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        
        self.rank = dist.get_rank()
        
        self.world_size = dist.get_world_size()
        
        self._params: List[nn.Parameter] = [p for _, p in module.named_parameters()]
        
        self._buffers = [b for _, b in module.named_buffers()]
        
        self.buckets: List[Dict[str, Any]] = []
        
        self._hook_handles = []
        
        self._broadcast_parameters_and_buffers()
        
        self._build_buckets()
        
        self._register_hooks()
        
    def _broadcast_parameters_and_buffers(self):
        seen_params: Set[int] = set()
        with torch.no_grad():
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
    
    def _create_buckets(self):
        raise NotImplementedError
    
    def _build_buckets(self):
        raise NotImplementedError
    
    def _all_reduce_bucket(self):
        raise NotImplementedError
    
    def _register_hooks(self):
        seen_params: Set[int] = set()
        for p in self._params:
            if id(p) in seen_params or not p.requires_grad:
                continue
            seen_params.add(id(p))
            
            def make_hook(param):
                def hook(grad):
                    if grad is None:
                        return
                    
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                    
                    grad.div_(self.world_size)
                    
                    return grad
                return hook
            
            handle = p.register_hook(make_hook(p))
            self._hook_handles.append(handle)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        raise NotImplementedError


def get_device(index: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")