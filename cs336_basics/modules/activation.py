from utils.core_imports import (
    os, math, jaxtyping, torch, Tensor, Optimizer,
    Module, ModuleList, Parameter, sigmoid,
    rearrange, einsum
)
from typing import Optional

__all__ = [
    "GLU",
    "Softmax"
]


class GLU(Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, input: Tensor) -> Tensor:
        return torch._C._nn.glu(input, self.dim)

# swish function equal silu 
def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class Softmax(Module):
    __constants__ = ["dim"]
    dim: Optional[int]
    
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, input: Tensor) -> Tensor:
        return torch.softmax(input, dim=-1)