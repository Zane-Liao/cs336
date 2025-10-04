import math
from typing import Optional
from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor
import triton
import triton.language as tl
from cs336_basics.modules import Linear, RotaryPositionalEmbedding

__all__ = [
    "FlashAttention",
    "FlashAttnAutogradFunction",
    "TritonFlashAttentionAutogradFunction",
]


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


class FlashAttnAutogradFunction(torch.autograd.Function):
    """Naive Pytorch Implementation"""
    @staticmethod
    def forward(
        ctx,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        is_causal=False
        ):
        B, N, d = Q.shape
        
        # Define block_size
        B_q, B_k = 32, 32
        
        # Compute
        T_q = (N + B_q - 1) // B_q
        T_k = (N + B_k - 1) // B_k
        
        # $O_i = \sum_j \text{softmax}(S_{ij}) V_j$
        O = torch.zeros_like(Q)
        # $L_i = \log\left(\sum_j \exp(S_{ij})\right)$
        L = torch.zeros((B, N), device=Q.device)
        
        for b in range(B):
            for i in range(T_q):
                start_q = i * B_q
                end_q = min((i + 1) * B_q, N)
            
                # Load Q_i from global memory
                Q_i = Q[b, start_q : end_q, :]
            
                # Initialize O_i l_i m_i
                O_i = torch.zeros_like(Q_i)
                l_i = torch.zeros((Q_i.shape[0], 1))
                # -float('inf') ==> −∞
                m_i = torch.full((Q_i.shape[0], 1), float('-inf'), device=Q.device)

                for j in range(T_k):
                    start_k = j * B_k
                    end_k = min((j + 1) * B_k, N)
                
                    # Load K(j), V(j) from global memory
                    # K, V shape into ==> B_k x d
                    K_j = K[b, start_k : end_k, :]
                    V_j = V[b, start_k : end_k, :]
                
                    # mask
                    if is_causal and end_k > end_q:
                        break
                
                    # Compute tile of pre-softmax attention scores
                    # S_ij = (Q_i @ K_j.transpose(-2, -1)) / torch.sqrt(d)
                    S_ij = (Q_i @ K_j.T) / math.sqrt(d)
                
                    if is_causal:
                        row_idx = torch.arange(start_q, end_q, device=Q.device).view(-1, 1)
                        col_idx = torch.arange(start_k, end_k, device=Q.device).view(1, -1)
                    
                        mask = row_idx >= col_idx
                        S_ij = S_ij.masked_fill(~mask, float('-inf'))
                    
                    # Initial
                    m_ij = torch.max(S_ij, dim=1, keepdim=True).values
                    P_ij = torch.exp(S_ij - m_ij)
                    l_ij = torch.sum(P_ij, dim=1, keepdim=True)
                
                    m_new = torch.maximum(m_i, m_ij)
                
                    exp_m_diff = torch.exp(m_i - m_new)
                    exp_m_ij_diff = torch.exp(m_ij - m_new)
                
                    l_i = (exp_m_diff * l_i) + (exp_m_ij_diff * l_ij)
                
                    O_i = exp_m_diff * O_i
                    O_i = O_i + (exp_m_ij_diff * (P_ij @ V_j))
                
                    m_i = m_new
                    # End for
            
                O_i = O_i / l_i
                O[b, start_q : end_q, : ] = O_i
                L[b, start_q : end_q] = (m_i + torch.log(l_i)).view(-1)
                # End for
        
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        
        dQ, dK, dV = torch_flash_bwd(
            ctx, dO, Q, K, V, O, L, is_causal=is_causal
        )
        
        return dQ, dK, dV, None


class TritonFlashAttentionAutogradFunction(torch.autograd.Function):
    """Triton Implementation"""
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.shape[-1] == K.shape[-1] == V.shape[-1]
        assert Q.ndim == 4 and K.ndim == 4 and V.ndim == 4
        assert Q.is_cuda and K.is_cuda and V.is_cuda

        B, H, S, d = Q.shape
        _, _, S_k, _ = K.shape

        Q_flat = Q.view(B * H, S, d).contiguous()
        K_flat = K.view(B * H, S_k, d).contiguous()
        V_flat = V.view(B * H, S_k, d).contiguous()

        O_flat = torch.zeros_like(Q_flat)
        L_flat = torch.zeros((B * H, S), device=Q.device, dtype=torch.float32)

        scale = 1.0 / (d ** 0.5)
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32

        grid = (triton.cdiv(S, Q_TILE_SIZE), B * H)

        flash_fwd_kernel[grid](
            Q_ptr=Q_flat, K_ptr=K_flat, V_ptr=V_flat,
            O_ptr=O_flat, L_ptr=L_flat,
            stride_qb=Q_flat.stride(0), stride_qq=Q_flat.stride(1), stride_qd=Q_flat.stride(2),
            stride_kb=K_flat.stride(0), stride_kk=K_flat.stride(1), stride_kd=K_flat.stride(2),
            stride_vb=V_flat.stride(0), stride_vk=V_flat.stride(1), stride_vd=V_flat.stride(2),
            stride_ob=O_flat.stride(0), stride_oq=O_flat.stride(1), stride_od=O_flat.stride(2),
            stride_lb=L_flat.stride(0), stride_lq=L_flat.stride(1),
            N_QUERIES=S, N_KEYS=S_k, scale=scale, D=d,
            Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE,
            is_causal=bool(is_causal)
        )

        O = O_flat.view(B, H, S, d)
        ctx.save_for_backward(Q, K, V, O, L_flat)
        ctx.is_causal = bool(is_causal)

        return O
    
    @staticmethod
    def backward(ctx, dO):
        """OPTIONAL, Impl Triton backward"""
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        
        dQ, dK, dV = torch_flash_bwd(
            ctx, dO, Q, K, V, O, L, is_causal=is_causal
        )
        
        return dQ, dK, dV, None


def torch_flash_bwd(ctx, dO, Q, K, V, O, L, is_causal=None):
    B, N, d = Q.shape
        
    D = torch.sum(dO * O, dim=-1, keepdim=True)
        
    scale = 1.0 / math.sqrt(d)
        
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    # Define block_size
    B_q, B_k = 32, 32
        
    # Compute
    T_q = (N + B_q - 1) // B_q
    T_k = (N + B_k - 1) // B_k
        
    for b in range(B):
        for j in range(T_k):
            start_k = j * B_k
            end_k = min((j + 1) * B_k, N)
                
            K_j = K[b, start_k : end_k, :]
            V_j = V[b, start_k : end_k, :]
                
            dK_j = torch.zeros_like(K_j)
            dV_j = torch.zeros_like(V_j)
                
            for i in range(T_q):
                start_q = i * B_q
                end_q = min((i + 1)  * B_q, N)
                    
                Q_i = Q[b, start_q : end_q, :]
                O_i = O[b, start_q : end_q, :]
                dO_i = dO[b, start_q : end_q, :]
                # dQ_i = Q[b, start_q : end_q, :]
                L_i = L[b, start_q : end_q]
                D_i = D[b, start_q : end_q, :]
                    
                S_ij = Q_i @ K_j.T * scale
                    
                if is_causal:
                    row_idx = torch.arange(start_q, end_q, device=Q.device).view(-1, 1)
                    col_idx = torch.arange(start_k, end_k, device=Q.device).view(1, -1)
                    
                    mask = row_idx >= col_idx
                    S_ij = S_ij.masked_fill(~mask, float('-inf'))
                    
                P_ij = torch.exp(S_ij - L_i.unsqueeze(-1))
                    
                dV_j += P_ij.transpose(0, 1) @ dO_i

                dP_ij = dO_i @ V_j.transpose(0, 1)

                # sqrt(d) ==> scale
                dS_ij = P_ij * (dP_ij - D_i)
                    
                dQ_i = dS_ij @ K_j
                dQ_i = dQ_i * scale
                    
                # sqrt(d)
                dK_j += (dS_ij.transpose(0, 1) @ Q_i) * scale

                dQ[b, start_q : end_q, :] += dQ_i
                # End for

            dK[b, start_k : end_k, :] += dK_j
            dV[b, start_k : end_k, :] += dV_j
            # End for
        
    return dQ, dK, dV


# Triton Implement
@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    start_m = query_tile_index * Q_TILE_SIZE
    offs_m = start_m + tl.arange(0, Q_TILE_SIZE)
    mask_m = offs_m < N_QUERIES
    
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(start_m, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, ),
        strides=(stride_lq, ),
        offsets=(start_m, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, ),
    )
    
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Q_i = Q_i.to(tl.float32)

    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE, ), dtype=tl.float32)
    # float('-inf')
    m_i = tl.full((Q_TILE_SIZE, ), -1e9, dtype=tl.float32)
    
    for start_n in range(0, N_KEYS, K_TILE_SIZE):
        offs_n = start_n + tl.arange(0, K_TILE_SIZE)
        mask_n = offs_n < N_KEYS 
    
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        K_j = K_j.to(tl.float32)
        V_j = V_j.to(tl.float32)
        
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale
        
        combined_mask = (~mask_m[:, None]) | (~mask_n[None, :])
        if is_causal:
            causal_mask = (offs_n[None, :] > offs_m[:, None])
            combined_mask = combined_mask | causal_mask

        neg_inf = tl.full(S_ij.shape, -1e9, dtype=tl.float32)
        S_ij = tl.where(combined_mask, neg_inf, S_ij)
        
        m_ij = tl.max(S_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        exp_m_ij_diff = tl.exp(S_ij - m_new[:, None])
        l_ij = tl.sum(exp_m_ij_diff, axis=1)
        
        exp_m_diff = tl.exp(m_i - m_new)
        l_i = exp_m_diff * l_i + l_ij
        
        O_i = exp_m_diff[:, None] * O_i
        P_ij = tl.dot(exp_m_ij_diff.to(V_j.dtype), V_j)
        O_i += P_ij
        
        m_i = m_new
        
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    O_i = O_i / l_i_safe[:, None]
    
    tl.store(O_block_ptr, O_i.to(O_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, l_i, boundary_check=(0, ))