"""
----
train_model.py

Warning: This script is not applicable to Windows. 
If you need to use it, please modify it yourself.

My OS: Macos26(Linux)
Shell: zsh
Use argparse and yaml to configure the run
subprocess Create shell process

This file may be a redundant file, but if you need to run all at once
without step-by-step, you can try it.
Warning: Unpredictable errors may occur, so use with caution.
----
"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from .tokenizer import *

from utils.core_imports import (
    yaml, argparse, subprocess
)

import numpy as np
import numpy.typing as npt

from utils.core_imports import (
    math, jaxtyping, torch, Tensor, Optimizer,
    Module, ModuleList, Parameter, sigmoid,
    rearrange, einsum
)

from modules.layers import TransformerLM

from modules.loss import CrossEntropyLoss
from modules.optimizer import SGD, AdamW, compute_lr, gradient_cliping

# Solution
def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    starting_idxs = torch.randint(len(dataset) - context_length, (batch_size,))
    x = torch.stack([torch.from_numpy((dataset[i:i+context_length]).astype(np.int64)) for i in starting_idxs])
    y = torch.stack([torch.from_numpy((dataset[i+1:i+1+context_length]).astype(np.int64)) for i in starting_idxs])
    if "cuda" in device:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def load(src, model, optimizer):
    check_point = torch.load(src, map_location=torch.device('cpu'))
    
    model.load_state_dict(check_point['model_state_dict'])
    
    optimizer.load_state_dict(check_point['optimizer_state_dict'])
    
    iteration = check_point['iteration']
    
    print(f"load {src} iterations: {iteration}")
    
    # model.train()
    return iteration


def save(model, optimizer, iteration, out):
    check_point = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    torch.save(check_point, out) 
    print(f"save {out} iterations: {iteration}")


def train():
    raise NotImplementedError


if __name__ == '__main__':
    train()