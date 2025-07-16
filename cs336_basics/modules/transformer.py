import regex as re 
import math
import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Sequential, Parameter
from torch.optim import Optimizer
from .activation import (
    GeLU,
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


class DenseTransformerDecoder(Module):
    """"""
    def __init__():
        raise NotImplementedError
