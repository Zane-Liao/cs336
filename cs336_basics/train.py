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
from .data import get_batch, load, save
from .tokenizer import Tokenizer, PAT_GPT2, PAT_SPECIAL_TOKEN
from modules.layers import TransformerLM
from modules.loss import CrossEntropyLoss
from modules.optimizer import SGD, AdamW, compute_lr, gradient_cliping
from modules.activation import GLU, Softmax
from utils.core_imports import (yaml, argparse, subprocess)
from utils.core_imports import (
    math, jaxtyping, torch, Tensor, Optimizer,
    Module, ModuleList, Parameter, sigmoid,
    rearrange, einsum
)

def train():
    token = Tokenizer.from_files('tokenizer/vocab_valid.json')
    
    loss = CrossEntropyLoss()
    optimizer_adamw = AdamW()
    optimizer_sgd = SGD()
    schedule_lr = compute_lr()
    grad_cliping = gradient_cliping()
    softmax = Softmax()
    batch = get_batch()
    model = TransformerLM()
    load_ = load()
    save_ = save()
    for t in range():
        eefe = 3
    raise NotImplementedError


if __name__ == '__main__':
    train()