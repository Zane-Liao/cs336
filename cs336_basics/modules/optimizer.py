"""
---

---
"""
import math
import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Sequential, Parameter
from torch.optim import Optimizer
from collections.abc import Callable, Iterable
from typing import Optional

__all__ = [
    "SGD",
    "AdamW"
]


class SGD(Optimizer):
    """
    PDF version of SGD would be implemented as a PyTorch Optimizer:
    """
    def __init__(self, params, lr=1e-3):
        """
        should initialize your optimizer. Here, params will be a collection of
        parameters to be optimized (or parameter groups, in case the user 
        wants to use different hyperpa-rameters, such as learning rates, 
        for different parts of the model). Make sure to pass params to the
        __init__ method of the base class, which will store these parameters 
        for use in step. You can take additional arguments depending on the
        optimizer (e.g., the learning rate is a common one), and pass
        them to the base class constructor as a dictionary, where keys are
        the names (strings) you choose for these parameters.
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        """
        should make one update of the parameters. During the training loop,
        this will be called after the backward pass, so you have access to
        the gradients on the last batch. This method should iterate through
        each parameter tensor p and modify them in place, i.e. setting p.data,
        which holds the tensor associated with that parameter based on the 
        gradient p.grad (if it exists), the tensor representing the 
        gradient of the loss with respect to that parameter.
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.

        return loss


class Adam(Optimizer):
    """"""
    def __init__(self):
        raise NotImplementedError
    
    def forward(self):
        raise NotImplementedError

    
class AdamW(Adam):
    """
    Implement the AdamW optimizer as a subclass of torch.optim.Optimizer.
    Your class should take the learning rate α in __init__, as well as the β,
    ϵ and λ hyperparameters. To help you keep state, the base Optimizer class
    gives you a dictionary self.state, which maps nn.Parameter objects to
    a dictionary that stores any information you need for that parameter
    (for AdamW, this would be the moment estimates).
    """
    def __init__(self):
        raise NotImplementedError
    
    def forward(self):
        raise NotImplementedError
