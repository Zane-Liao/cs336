import regex as re 
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


class DenseTransformerDecoder(Module):
    """"""
    def __init__():
        raise NotImplementedError
