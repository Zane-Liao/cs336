from utils.core_imports import (
    math, jaxtyping, torch, Tensor, Optimizer,
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


class DenseTransformerDecoder(Module):
    """"""
    def __init__(self):
        raise NotImplementedError
    
    def forward(self):
        raise NotImplementedError
