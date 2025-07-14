# This code is adapted from:
# https://github.com/karpathy/minbpe/blob/master/minbpe/base.py
# Original author: Andrej karpathy
# Modifications: Adjusted data loader and optimizer for custom dataset.


class Tokenizer:
    """Abstract interface for a tokenizer."""
    def __init__(self):
        raise NotADirectoryError
    
    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError
    
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError
    
    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError
    
    def _build_vocab(self):
        raise NotImplementedError
    
    def save(self, file_prefix):
        raise NotImplementedError
    
    def load(self, model_file):
        raise NotImplementedError
    
    
def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  # @inspect indices, @inspect pair, @inspect new_index
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []  # @inspect new_indices
    i = 0  # @inspect i
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices