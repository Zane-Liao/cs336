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
    返回一个能在多进程间分片参数状态的优化器。
    每个 rank 只维护自己负责的那部分参数的优化器状态。
    
    Args:
        params: 模型参数（iterator）
        optimizer_cls: 优化器类（如 torch.optim.AdamW）
        **kwargs: 传递给优化器的参数（如 lr, weight_decay 等）
    
    Returns:
        优化器实例
    """
    if not dist.is_initialized():
        # 如果没有初始化分布式，就用普通优化器
        return optimizer_cls(params, **kwargs)
    
    # 获取分布式信息
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 将参数转换为列表（保存所有参数的引用）
    all_params = list(params)
    
    # 为每个参数分配所有者 rank（轮询方式）
    param_to_rank = {}
    for i, p in enumerate(all_params):
        owner_rank = i % world_size
        param_to_rank[id(p)] = owner_rank
    
    # 找出当前 rank 拥有的参数
    local_params = [p for p in all_params if param_to_rank[id(p)] == rank]
    
    # 只用本地参数初始化底层优化器
    local_optimizer = optimizer_cls(local_params, **kwargs)
    
    # 创建一个包装类来拦截 step() 方法
    class ShardedOptimizerWrapper:
        def __init__(self, local_opt, all_params_list, param_rank_map, world_sz):
            self.local_optimizer = local_opt
            self.all_params = all_params_list
            self.param_to_rank = param_rank_map
            self.world_size = world_sz
        
        def zero_grad(self, set_to_none: bool = False):
            """清空所有参数的梯度"""
            for p in self.all_params:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.detach_()
                        p.grad.zero_()
        
        @torch.no_grad()
        def step(self, closure=None):
            """执行优化步骤"""
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            
            # 步骤 1: 梯度聚合 (all-reduce)
            for p in self.all_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(self.world_size)
            
            # 步骤 2: 本地更新（只更新自己拥有的参数）
            self.local_optimizer.step()
            
            # 步骤 3: 参数广播（从所有者同步到其他 rank）
            for p in self.all_params:
                owner_rank = self.param_to_rank[id(p)]
                dist.broadcast(p.data, src=owner_rank)
            
            return loss
        
        def state_dict(self):
            """返回优化器状态"""
            return {
                'local_optimizer': self.local_optimizer.state_dict(),
                'param_to_rank': self.param_to_rank,
            }
        
        def load_state_dict(self, state_dict):
            """加载优化器状态"""
            self.local_optimizer.load_state_dict(state_dict['local_optimizer'])
            self.param_to_rank = state_dict['param_to_rank']
        
        def __getattr__(self, name):
            """将其他属性访问转发到本地优化器"""
            return getattr(self.local_optimizer, name)
    
    return ShardedOptimizerWrapper(local_optimizer, all_params, param_to_rank, world_size)