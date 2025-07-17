import math
import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Parameter
from torch.optim import Optimizer


__all__ = [
    "CrossEntropyLoss"
]


class Loss(Module):
    """"""
    def __init__():
        raise NotImplementedError
    

class WeightedLoss(Loss):
    """"""
    def __init__():
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
    def __init__():
        raise NotImplementedError
    
    def step(self):
        raise NotImplementedError
    
