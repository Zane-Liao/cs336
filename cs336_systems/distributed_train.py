"""
---
2025.10
OS: Ubuntu Linux
GPU: GPU PCIE X 4
device: GPU0 GPU1 GPU2 GPU3
Shell: bash
Improvement: torch.compile() FlashAttention DDP OptimizerStateShare
Our Impl: Use torch.nn.parallel.DistributedDataParallel
"""

