from utils.core_imports import (
    os, math, np, jaxtyping, torch, init, Tensor, Optimizer,
    Module, ModuleList, Parameter, Optional,
    sigmoid, rearrange, einsum
)

from .activation import GLU, Softmax, silu, scaled_dot_product_attention

__all__ = [
    "Embedding",
    "Linear",
    "RMSNorm",
    "SwiGLU",
    "RotaryPositionalEmbedding",
    "ScaledDotProductAttention",
    "MultiHeadSelfAttention",
    "TransformerBlock",
    "TransformerLM"
]


class Linear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        linear transformation module.
    
        Parameters:
            in_features: int final dimension of the input
        
            out_features: int final dimension of the output
        
            device: torch.device | None = None Device to store the parameters on
        
            dtype: torch.dtype | None = None Data type of the parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        std = math.sqrt(2 / (in_features + out_features))
        init.trunc_normal_(
            self.weight, mean=0.0, std=std, a=-3.0*std, b=3.0*std
            )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the linear transformation to the input.
        
        Parameter:
            x: torch.Tensor
        Return:
            torch.Tensor
        """
        output = x @ self.weight.T
        return output


class Embedding(Module):
    __constants__ = ['num_embeddings', 'embedding_dim']
    num_embeddings: int
    embedding_dim: int
    weight: Tensor
    
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        an embedding module.
        
        Parameters:
            num_embeddings: int Size of the vocabulary
            
            embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
            
            device: torch.device | None = None Device to store the parameters on
            
            dtype: torch.dtype | None = None Data type of the parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        )
        init.trunc_normal_(
            self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0
        )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        
        Parameter:
            token_ids: torch.Tensor
        Return:
            torch.Tensor
        """
        return torch.embedding(self.weight, token_ids)


# I refer to torch concise implementation
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
        super().__init__()
        self.eps = eps
        self.weight = Parameter(torch.ones(d_model, device=device, dtype=dtype))
        
    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape.
        
        Parameter:
            x: torch.Tensor
        Return:
            torch.Tensor
        """
        # In torch, x.float() => float32
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# position-wise feed-forward network
# Come from pytorch PR: Support Swiglu for Module and functional 144465
class SwiGLU(Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff) 
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


# I referred to some online Solution
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
        super().__init__()
        half_dim = d_k // 2
        self.theta = (1. / theta ** (torch.arange(0, half_dim, device=device) / half_dim))
        
        self.max_seq_len = torch.arange(max_seq_len, device=device)
        angle_max_seq_len = torch.einsum("i,j->ij", self.max_seq_len, self.theta)

        self.register_buffer("sin", torch.sin(angle_max_seq_len), persistent=False)
        self.register_buffer("cos", torch.cos(angle_max_seq_len), persistent=False)
    
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
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]
        
        x_even = x[..., 0::2] 
        x_odd = x[..., 1::2]
        
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        
        return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)


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
        super().__init__()

    def forward(self, query, key, value, mask):
        return scaled_dot_product_attention(query, key, value, mask)


class MultiHeadSelfAttention(Module):
    """
    Implement causal multi-head self-attention as a torch.nn.Module.
        We have (batch_size, seq_len, d_model)
        reshape (batch_size, seq_len, h, d_k)
        d_model == h * d_k ⟹ d_k = d_model // h
        
    Parameters:
        d_model: int Dimensionality of the Transformer block inputs

        num_heads: int Number of heads to use in multi-head self-attention

        Datermine whether rope_exist: (RotaryPositionalEmbedding exists) true or false
        
        theta: parameter for RotaryPositionalEmbedding

        max_seq_len: parameter for RotaryPositionalEmbedding
        
        token_positions: parameter for RotaryPositionalEmbedding
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.qkv_proj = Linear(d_model, 3 * d_model)
        self.o_proj = Linear(d_model, d_model)

    def forward(self,
                in_features: Tensor,
                rope_exist: bool = False,
                theta: float | None = None,
                max_seq_len: int | None = None, 
                token_positions: Tensor | None = None
                ):
        d_k = self.d_model // self.num_heads
        self.rope_exist = rope_exist
        self.rope: Optional[RotaryPositionalEmbedding] = (
            RotaryPositionalEmbedding(theta, d_k, max_seq_len)
            if rope_exist else None
            )
        
        batch_size, seq_len, _ = in_features.shape

        qkv = self.qkv_proj(in_features)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.num_heads)

        if self.rope_exist is True:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        casual_mask = torch.triu(torch.ones(1, 1, seq_len, seq_len), diagonal=1).bool()
        output = scaled_dot_product_attention(q, k, v, torch.logical_not(casual_mask))
        
        return self.o_proj(rearrange(output, "b h t d -> b t (h d)"))


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
