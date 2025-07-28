# prepare_data.py
import os
import numpy as np
from tokenizer import Tokenizer

def tokenize_and_save(input_txt_path, output_bin_path, tokenizer):
    with open(input_txt_path, 'r', encoding='utf-8') as f:
        text_data = f.read()

    tokens = tokenizer.encode(text_data)
    
    tokens_np = np.array(tokens, dtype=np.uint16)

    tokens_np.tofile(output_bin_path)

if __name__ == '__main__':
    vocab_path = 'tokenizer/vocab.json'
    merges_path = 'tokenizer/merges.txt'
    
    input_train_txt_path = 'data/train.txt'
    input_val_txt_path = 'data/val.txt'

    output_train_bin_path = 'data/train.bin'
    output_val_bin_path = 'data/val.bin'

    print("Begin...")

    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path
    )
    
    tokenize_and_save(input_train_txt_path, output_train_bin_path, tokenizer)
    
    tokenize_and_save(input_val_txt_path, output_val_bin_path, tokenizer)
    
    print("Finish!")