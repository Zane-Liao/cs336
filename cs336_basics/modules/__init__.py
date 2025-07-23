"""
---

---
"""
import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .activation import (
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

__all__ = [
    "SiLU",
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
    "DenseTransformerDecoder",
]