from __future__ import annotations

from typing import Type

import torch
import torch.distributed as dist

# from cs336_systems.flash_attention import FlashAttnAutogradFunction, TritonFlashAttentionAutogradFunction
from cs336_systems.ddp_model import DDPIndividualParameters, BucketDDPIndividualParameters
from cs336_systems.optimizer_share import OptimizerStateShare


def get_flashattention_autograd_function_pytorch() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyFlashAttnAutogradFunctionClass
    # return FlashAttnAutogradFunction


def get_flashattention_autograd_function_triton() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_flashattention_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and backward passes.

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyTritonFlashAttentionAutogradFunctionClass
    # return TritonFlashAttentionAutogradFunction


def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    # For example: return DDPIndividualParameters(module)
    return DDPIndividualParameters(module)


def ddp_individual_parameters_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    # ddp_model.finish_gradient_synchronization()


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating buckets of gradients as they are ready
    in the backward pass.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
        bucket_size_mb: The bucket size, in megabytes. If None, use a single
            bucket of unbounded size.
    Returns:
        Instance of a DDP class.
    """
    return BucketDDPIndividualParameters(module, bucket_size_mb)


def ddp_bucketed_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    raise NotImplementedError


def ddp_bucketed_on_train_batch_start(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run at the very start of the training step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    raise NotImplementedError


def get_sharded_optimizer(params, optimizer_cls: type, **kwargs) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
        
    Returns:
        Instance of sharded optimizer.
    """
    if not dist.is_initialized():
        return optimizer_cls(params, **kwargs)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    all_params = list(params)

    param_to_rank = {}
    for i, p in enumerate(all_params):
        owner_rank = i % world_size
        param_to_rank[id(p)] = owner_rank
    
    local_params = [p for p in all_params if param_to_rank[id(p)] == rank]

    local_optimizer = optimizer_cls(local_params, **kwargs)

    class ShardedOptimizerWrapper:
        def __init__(self, local_opt, all_params_list, param_rank_map, world_sz):
            self.local_optimizer = local_opt
            self.all_params = all_params_list
            self.param_to_rank = param_rank_map
            self.world_size = world_sz
        
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

            for p in self.all_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(self.world_size)

            self.local_optimizer.step()

            for p in self.all_params:
                owner_rank = self.param_to_rank[id(p)]
                dist.broadcast(p.data, src=owner_rank)
            
            return loss
        
        def state_dict(self):
            return {
                'local_optimizer': self.local_optimizer.state_dict(),
                'param_to_rank': self.param_to_rank,
            }
        
        def load_state_dict(self, state_dict):
            self.local_optimizer.load_state_dict(state_dict['local_optimizer'])
            self.param_to_rank = state_dict['param_to_rank']
        
        def __getattr__(self, name):
            return getattr(self.local_optimizer, name)
    
    return ShardedOptimizerWrapper(local_optimizer, all_params, param_to_rank, world_size)