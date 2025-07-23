"""
---

---
"""
from utils.core_imports import (
    os, math, jaxtyping, torch, Tensor, Optimizer,
    Module, ModuleList, Parameter, sigmoid,
    rearrange, einsum
)

from .activation import (
    GLU,
    Softmax
)

from .layers import (
    Embedding,
    Linear,
    RMSNorm,
    SwiGLU,
    RotaryPositionalEmbedding,
    ScaledDotProductAttention,
    MultiHeadSelfAttention,
    TransformerBlock,
    TransformerLM
)

from .loss import CrossEntropyLoss
from .optimizer import SGD, AdamW

def train_transformer():
    """"""
    raise NotImplementedError


if __name__ == 'train_transformer':
    train_transformer()