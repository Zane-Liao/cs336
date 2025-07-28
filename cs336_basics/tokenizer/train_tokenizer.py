"""
Use tokenizers for comparison
"""
import os
import json
from tokenizer import train_bpe, PAT_SPECIAL_TOKEN

input_corpus_path = "../../data/TinyStoriesV2-GPT4-valid.txt"

vocab_size = 1268

output_dir = 'vocab'
output_voacb_path = os.path.join(output_dir, 'vocab_train.json')
output_merges_path = os.path.join(output_dir, 'merges.txt')

if __name__ == '__main__':
    print("Begin...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    vocab, merges = train_bpe(
        input_path=input_corpus_path,
        vocab_size=vocab_size,
        special_tokens=PAT_SPECIAL_TOKEN
    )
    
    print("Finish")
    
    decoded_vocab = { k: v.decode('utf-8', errors='replace') for k, v in vocab.items() }
    with open(output_voacb_path, 'w', encoding='utf-8') as f:
        json.dump(decoded_vocab, f, ensure_ascii=False, indent=2)
        
    print("merges...")
    with open(output_merges_path, 'w', encoding='utf-8') as f:
        f.write("#BPE Merges\n")
        for byte_1, byte_2 in merges:
            token1_str = byte_1.decode('utf-8', errors='replace')
            token2_str = byte_2.decode('utf-8', errors='replace')
            f.write(f"{token1_str} {token2_str}\n")

    print("Finish")