"""
---

---
"""
import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .activation import (
    GeGLU,
    SwiGLU,
    Softmax,
    LogSoftmax
)

from .layers import (
    TokenEmbedding,
    Linear,
    RMSNorm,
    Dropout,
    MultiheadAttention,
    RoPE
)

from .loss import CrossEntropyLoss
from .optimizer import AdamW
from .transformer import DenseTransformerDecoder