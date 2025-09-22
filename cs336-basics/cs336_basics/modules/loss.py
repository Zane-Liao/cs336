from typing import Optional
import torch
from torch import Tensor
from torch.nn import Module, _reduction as _Reduction

__all__ = [
    "CrossEntropyLoss",
]


class _Loss(Module):
    reduction: str
    
    # parameter reduction "mean" "sum" "none"
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean"):
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(
        self,
        weight: Optional[Tensor]=None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)
        self.weight: Optional[Tensor]


class CrossEntropyLoss(_WeightedLoss):
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
    def __init__(
        self,
        weight: Optional[Tensor]=None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        input: Tensor,
        target: Tensor,
    ) -> Tensor:
        return torch._C._nn.cross_entropy_loss(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=_Reduction.get_enum(self.reduction),
            label_smoothing=self.label_smoothing
        )