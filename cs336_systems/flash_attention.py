import math
import numpy as np
from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor
# import triton
# import triton.language as tl
from torch.profiler import ProfilerActivity
from torch.utils.cpp_extension import load_inline

__all__ = [
    "FlashAttnAutogradFunction",
    "TritonFlashAttentionAutogradFunction",
]


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
                    S_ij = torch.einsum("q d, k d -> q k", Q_i, K_j) / math.sqrt(d)
                
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
        B, N, d = Q.shape
        
        D = torch.sum(dO @ O, dim=-1)
        
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        # Define block_size
        B_q, B_k = 32, 32
        
        # Compute
        T_q = (N + B_q - 1) // B_q
        T_k = (N + B_k - 1) // B_k
        
        for b in range(B):
            for j in range(T_q):
                start_k = j * B_k
                end_k = min((j + 1) * B_k, N)
                
                K_j = K[b, start_k : end_k, :]
                V_j = K[b, start_k : end_k, :]
                
                dK_j = torch.zeros_like(K_j)
                dV_j = torch.zeros_like(V_j)
                
                for i in range(T_k):
                    start_q = i * B_q
                    end_q = min((i + 1)  * B_q, N)
                    
                    Q_i = Q[b, start_q : end_q, :]
                    O_i = O[b, start_q : end_q, :]
                    dO_i = dO[b, start_q : end_q, :]
                    dQ_i = Q[b, start_q : end_q, :]
                    
                    S_ij = torch.einsum("q d, k d -> q k", Q_i, K_j) / math.sqrt(d)
                    
                    P_ij = torch.exp(S_ij - L)
                    
                    dV_j += P_ij.transpose(0, 1) @ dO_i
                    dP_ij = dO_i @ V_j.transpose(0, 1)
                    dS_ij = P_ij @ (dP_ij - D) / math.sqrt(d)
                    
                    dQ_i += dS_ij @ K_j
                    dK_j += dS_ij.transpose(0, 1) @ Q_i
                    # End for

                dK[b, start_k : end_k, :] += dK_j
                dV[b, start_k : end_k, :] += dV_j
        
        return dQ, dK, dV, None


class TritonFlashAttentionAutogradFunction(torch.autograd.Function):
    """Triton Implementation"""
    @staticmethod
    def forward(self):
        raise NotImplementedError
    
    @staticmethod
    def backward(self):
        raise NotImplementedError


# # Triton Implement
# @triton.jit
# def flash_fwd_kernel(
#     Q_ptr,
#     K_ptr,
#     V_ptr,
#     O_ptr,
#     L_ptr,
#     stride_qb,
#     stride_qq,
#     stride_qd,
#     stride_kb,
#     stride_kk,
#     stride_kd,
#     stride_vb,
#     stride_vk,
#     stride_vd,
#     stride_ob,
#     stride_oq,
#     stride_od,
#     stride_lb,
#     stride_lq,
#     N_QUERIES,
#     N_KEYS,
#     scale,
#     D: tl.constexpr,
#     Q_TILE_SIZE: tl.constexpr,
#     K_TILE_SIZE: tl.constexpr,
# ):
#     raise NotImplementedError


# @triton.jit
# def flash_bwd_kernel(
#     Q_ptr,
#     K_ptr,
#     V_ptr,
#     O_ptr,
#     L_ptr,
#     DQ_ptr,
#     DK_ptr,
#     DV_ptr,
#     DO_ptr,
#     stride_qb,
#     stride_qq,
#     stride_qd,
#     stride_kb,
#     stride_kk,
#     stride_kd,
#     stride_vb,
#     stride_vk,
#     stride_vd,
#     stride_ob,
#     stride_oq,
#     stride_od,
#     stride_lb,
#     stride_lq,
#     N_QUERIES,
#     N_KEYS,
#     scale,
#     D: tl.constexpr,
#     Q_TILE_SIZE: tl.constexpr,
#     K_TILE_SIZE: tl.constexpr,
# ):
#     raise NotImplementedError