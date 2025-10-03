import timeit
import itertools
from typing import Callable, Optional
from einops import rearrange
import torch
from cs336_basics.modules import Linear, RotaryPositionalEmbedding
from cs336_basics.modules import AdamW
from cs336_systems.benchmark_script.profiling_benchmark import *
from flash_attention import *


class FlashAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 theta: float | None = None,
                 max_seq_len: int | None = None,
                 rope_exist: bool | None = None,
                 device=None,
                 dtype=None,
            ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.qkv_proj = Linear(d_model, 3 * d_model, **factory_kwargs)
        self.o_proj = Linear(d_model, d_model, **factory_kwargs)
        
        self.rope_exist = rope_exist
        if self.rope_exist:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device
            )
        else:
            self.rope = None

    def forward(self,
                in_features: Tensor,
                token_positions: Optional[Tensor] = None,
               ) -> Tensor:
        
        batch_size, seq_len, _ = in_features.shape

        qkv = self.qkv_proj(in_features)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.num_heads)

        if self.rope_exist:
            if token_positions is None:
                raise ValueError("token_positions must be provided when use_rope is True.")
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        # output = TritonFlashAttentionAutogradFunction.apply(q, k, v)
        output = torch.compile(FlashAttnAutogradFunction.apply(q, k, v))
        
        return self.o_proj(rearrange(output, "b h t d -> b t (h d)"))


def flash_benchmarking():
    raise NotImplementedError