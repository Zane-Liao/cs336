from utils.core_imports import (
    os, math, jaxtyping, torch, Tensor, Optimizer,
    Module, ModuleList, Parameter, sigmoid,
    rearrange, einsum
)

__all__ = [
    "CrossEntropyLoss"
]


class Loss(Module):
    """"""
    def __init__(self):
        raise NotImplementedError
    

class WeightedLoss(Loss):
    """"""
    def __init__(self):
        raise NotImplementedError
    

class CrossEntropyLoss(WeightedLoss):
    """
    Deliverable: Write a function to compute the cross entropy loss,
    which takes in predicted logits (oi) and targets (xi+1) and computes
    the cross entropy ℓi =−log softmax(oi)[xi+1].
        
    • Subtract the largest element for numerical stability.
    • Cancel out log and exp whenever possible.
    • Handle any additional batch dimensions and return the average 
    across the batch. As with sec-tion 3.3, 
    we assume batch-like dimensions always come first,
    before the vocabulary size dimension.
    """
    def __init__(self):
        raise NotImplementedError
    
    def step(self):
        raise NotImplementedError
    
