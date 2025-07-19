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
    SiLU,
    Softmax
)

from .layers import (
    Embedding,
    Linear,
    RMSNorm,
    GLU,
    SwiGLU,
    RotaryPositionalEmbedding,
    ScaledDotProductAttention,
    MultiHeadSelfAttention,
    TransformerBlock,
    TransformerLM
)

from .loss import CrossEntropyLoss
from .optimizer import SGD, AdamW
from .transformer import DenseTransformerDecoder


def train_transformer():
    """"""
    raise NotImplementedError


if __name__ == 'train_transformer':
    train_transformer()