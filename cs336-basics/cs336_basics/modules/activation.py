import math
from typing import Optional
import torch
from torch import Tensor
from torch.nn import Module
from typing import Optional

__all__ = [
    "GLU",
    "Softmax",
]

# swish function equal silu 
def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)

# cal attention-score
def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor, 
    value: Tensor, 
    mask: Tensor | None = None,
) -> Tensor:
    d_k = key.shape[-1]
    # scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # We use torch.einsum, not code: from einops import einsum
    scores = torch.einsum("... q d, ... k d -> ... q k", query, key) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    softmax = Softmax()
    return softmax(scores) @ value

class GLU(Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, input: Tensor) -> Tensor:
        return torch._C._nn.glu(input, self.dim)


class Softmax(Module):
    __constants__ = ["dim"]
    dim: Optional[int]
    
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, input: Tensor) -> Tensor:
        return torch.softmax(input, dim=-1)