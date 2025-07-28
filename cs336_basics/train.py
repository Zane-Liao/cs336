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
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import yaml
from .data import get_batch, load, save
from .tokenizer import Tokenizer, PAT_GPT2, PAT_SPECIAL_TOKEN
from modules.layers import TransformerLM
from modules.loss import CrossEntropyLoss
from modules.activation import GLU, Softmax
from modules.optimizer import SGD, AdamW, compute_lr, gradient_cliping

def train():
    # 1. Load Config
    print("--- Loading Configuration ---")
    with open('config.yaml', 'r') as f:
        load = yaml.safe_load(f)
        
    model_args = load['model_args']
    training_args = load['training_args']
    data_args = load['data_args']
    
    os.makedirs(data_args['checkpoint_dir'], exist_ok=True)
    
    # 2. Initial
    print("--- Initializing ---")
    device = torch.device(training_args['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = Tokenizer.from_files(
        vocab_filepath=data_args['vocab_path'],
        merges_filepath=data_args['merges_path']
    )
    
    model_args['vocab_size'] = len(tokenizer.vocab)
    print(f"Tokenizer loaded. Vocab size: {model_args['vocab_size']}")
    
    model = TransformerLM(**model_args).to(device)
    print(f"Model created with {model.get_num_params():,} parameters.")
    
    optimizer = AdamW(model.parameters(), lr=training_args['learning_rate'])
    
    loss_init = CrossEntropyLoss()
    
    # 3. Load data
    print("--- Loading Data with np.memmap ---")
    train_data = np.memmap(data_args['train_data_path'], dtype=np.uint16, model='r')
    valid_data = np.memmap(data_args['valid_data_path'], dtype=np.uint16, model='r')
    print(f"Train data tokens: {len(train_data):,}, Val data tokens: {len(valid_data):,}")

    # 4. Resume training
    start_iter = 0
    if data_args['resume_from_checkpoint']:
        print(f"Resuming training from {data_args['resume_from_checkpoint']}")
        start_iter = load(data_args['resume_from_checkpoint'], model, optimizer)
    
    # 5. Evaluation Function
    @torch.no_grad()
    def evaluate():
        model.eval()
        valid_loss = 0
        eval_iters = 100
        for _ in range(eval_iters):
            x, y = get_batch(valid_data, training_args['batch_size'], model_args['context_length'], device)
            logits = model(x)
            loss = loss_init(logits.view(-1, model_args['vocab_size']), y.view(-1))
            valid_loss += loss.item()
        model.train()
        return valid_loss / eval_iters  

    # 6. Begin Training Loop
    print("--- Starting Training Loop ---")
    t0 = time.time()
    for iter_num in range(start_iter, training_args['max_iters']):
        
        lr = compute_lr(
            iter_num, 
            training_args['learning_rate'],
            training_args['min_lr'],
            training_args['warmup_steps'],
            training_args['lr_decay_steps']
        )
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        inputs, targets = get_batch(train_data, training_args['batch_size'], model_args['context_length'], device)
        
        logits = model(inputs)
        loss = loss_init(logits.view(-1, model_args['vocab_size']), targets.view(-1))
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        gradient_cliping(model.parameters(), training_args['gradient_clip_val'])
        
        optimizer.step()
        

        if iter_num % 10 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"Iter {iter_num}/{training_args['max_iters']}, Train Loss: {loss.item():.4f}, LR: {lr:.6f}, Time: {dt*1000:.2f}ms")

        if iter_num > 0 and iter_num % training_args['eval_interval'] == 0:
            val_loss = evaluate()
            print(f"--- Eval at iter {iter_num}: Val Loss: {val_loss:.4f} ---")
            
            checkpoint_path = os.path.join(data_args['checkpoint_dir'], f"model_iter_{iter_num}.pt")
            save(model, optimizer, iter_num, checkpoint_path)

    print("--- Training Finished! ---")
    final_checkpoint_path = os.path.join(data_args['checkpoint_dir'], "model_final.pt")
    save(model, optimizer, training_args['max_iters'], final_checkpoint_path)
            
if __name__ == '__main__':
    train()