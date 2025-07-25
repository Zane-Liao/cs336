import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

__all__ = [
    "compute_lr",
    "gradient_cliping",
    "silu",
    "Softmax",
    "Embedding",
    "Linear",
    "RMSNorm",
    "GLU",
    "SwiGLU",
    "RotaryPositionalEmbedding",
    "ScaledDotProductAttention",
    "MultiHeadSelfAttention",
    "TransformerBlock",
    "TransformerLM",
    "CrossEntropyLoss",
    "SGD", 
    "AdamW",
]

from .activation import (
    GLU, Softmax, silu
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
from .optimizer import SGD, AdamW, compute_lr, gradient_cliping