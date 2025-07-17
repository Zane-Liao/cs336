import math
import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Parameter
from torch.optim import Optimizer
from torch import sigmoid
from .activation import SiLU, Softmax

__all__ = [
    "Embedding",
    "Linear",
    "RMSNorm",
    "GLU",
    "SwiGLU",
    "RotaryPositionalEmbedding",
    "ScaledDotProductAttention",
    "MultiHeadSelfAttention",
    "TransformerBlock",
    "TransformerLM"
]


class Linear(Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        linear transformation module.
    
        Parameters:
            in_features: int final dimension of the input
        
            out_features: int final dimension of the output
        
            device: torch.device | None = None Device to store the parameters on
        
            dtype: torch.dtype | None = None Data type of the parameters
        """
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        
        Parameter:
            x: torch.Tensor
        Return:
            torch.Tensor
        """
        raise NotImplementedError
    
class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        an embedding module.
        
        Parameters:
            num_embeddings: int Size of the vocabulary
            
            embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
            
            device: torch.device | None = None Device to store the parameters on
            
            dtype: torch.dtype | None = None Data type of the parameters
        """
        raise NotImplementedError
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        
        Parameter:
            token_ids: torch.Tensor
        Return:
            torch.Tensor
        """
        raise NotImplementedError


class RMSNorm(Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module.
        
        Parameters:
            d_model: int Hidden dimension of the model
        
            eps: float = 1e-5 Epsilon value for numerical stability
        
            device: torch.device | None = None Device to store the parameters on
        
            dtype: torch.dtype | None = None Data type of the parameters
        """
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape.
        
        Parameter:
            x: torch.Tensor
        Return:
            torch.Tensor
        """
        raise NotImplementedError


class GLU(Module):
    def __init__(self, input_dim, output_dim):
        """"""
        raise NotImplementedError
    
    def forward(self):
        """"""
        raise NotImplementedError
    

class SwiGLU(Module):
    def __init__(self):
        """"""
        raise NotImplementedError
    
    def forward(self):
        """"""
        raise NotImplementedError


class RotaryPositionalEmbedding(Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        RoPE module and create buffers if needed.
        
        Parameters:
            theta: float Θ value for the RoPE
            
            d_k: int dimension of query and key vectors
            
            max_seq_len: int Maximum sequence length that will be inputted
            
            device: torch.device | None = None Device to store the buffer on
        """
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return
        a tensor of the same shape. Note that you should tolerate
        x with an arbitrary number of batch dimensions. You should assume
        that the token positions are a tensor of shape (..., seq_len)
        specifying the token positions of x along the sequence dimension.
        
        Parameters:
            x: torch.Tensor
            token_positions: torch.Tensor
        Return:
            torch.Tensor
        """
        raise NotImplementedError


class ScaledDotProductAttention(Module):
    """
    Your implementation should handle keys and queries of shape
    (batch_size, ..., seq_len, d_k) and values of shape
    (batch_size, ..., seq_len, d_v), where... represents any number of
    other batch-like dimensions (if provided). The implementation should
    return an output with the shape (batch_size, ..., d_v).
    See section 3.3 for a discussion on batch-like dimensions.
    Your implementation should also support an optional user-provided boolean
    mask of shape (seq_len, seq_len). The attention probabilities of 
    positions with a mask value of True should collectively sum to 1,
    and the attention probabilities of positions with a mask value of False
    should be zero. To test your implementation against our provided tests,
    you will need to implement the test adapter at 
    [adapters.run_scaled_dot_product_attention].
    """
    def __init__(self):
        raise NotImplementedError
    
    def forward(self):
        raise NotImplementedError
    

class MultiHeadSelfAttention(Module):
    """
    Implement causal multi-head self-attention as a torch.nn.Module.
    
    Parameters:
        d_model: int Dimensionality of the Transformer block inputs.
        
        num_heads: int Number of heads to use in multi-head self-attention.
    """
    def __init__(self):
        raise NotImplementedError
    
    def forward(self):
        raise NotImplementedError


class TransformerBlock(Module):
    """
    Implement the pre-norm Transformer block as described in §3.5 and illustrated in Figure
        
    Parameters:
        d_model: int Dimensionality of the Transformer block inputs.
        
        num_heads: int Number of heads to use in multi-head self-attention.
        
        d_ff: int Dimensionality of the position-wise feed-forward inner layer.
    """
    def __init__(self):
        raise NotImplementedError
    
    def forward(self):
        raise NotImplementedError
    

class TransformerLM(Module):
    """
    Time to put it all together! Implement the Transformer language model
    as described in §3.1 and illustrated in Figure 1. At minimum,
    your implementation should accept all the aforementioned 
    construction parameters for the Transformer block.
    
    Parameters:
        vocab_size: embedding matrix. int The size of the vocabulary,
        necessary for determining the dimensionality of the token
        
        context_length: int The maximum context length, necessary for
        determining the dimensionality of the position embedding matrix.
        
        num_layers: int The number of Transformer blocks to use.
    """
    def __init__(self):
        raise NotImplementedError
    
    def forward(self):
        raise NotImplementedError
