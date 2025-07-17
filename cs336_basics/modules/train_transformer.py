"""
---

---
"""
import os
import random
import matplotlib
import math
import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Sequential, Parameter
from torch.optim import Optimizer
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