# This code is adapted from:
# https://github.com/karpathy/minbpe/blob/master/minbpe/base.py
# Original author: @karpathy
# Modifications: I modified some documents, the code is basically unchanged.

import regex
import random
import tiktoken
import unicodedata
import doctest
from typing import Iterable
# import sentencepiece

class Tokenizer:
    """Abstract interface for a tokenizer."""
    def __init__(self):
        raise NotADirectoryError
    
    # def train(self, text, vocab_size, verbose=False):
    #     # Tokenizer can train a vocabulary of size vocab_size from text
    #     raise NotImplementedError
    
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError
    
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        raise NotImplementedError
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        raise NotImplementedError
    
    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError
    
    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
    
    def save(self, file_prefix):
        raise NotImplementedError
    
    def load(self, model_file):
        raise NotImplementedError

def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: 
        >>> get_stats(a = [1, 2, 3, 1, 2])
        {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

    
def merge(
    indices: list[int],
    pair: tuple[int, int],
    new_index: int
) -> list[int]:  # @inspect indices, @inspect pair, @inspect new_index
    """Return `indices`, but with all instances of `pair` replaced with `new_index`.
    Example:
        >>> merge([5, 6, 6, 7, 9, 1], (6, 7), 99)
        [5, 6, 99, 9, 1]
    """
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


def get_compression_ratio(string: str, indices: list[int]) -> float:
    """Given `string` that has been tokenized into `indices`, ."""
    num_bytes = len(bytes(string, encoding="utf-8"))  # @inspect num_bytes
    num_tokens = len(indices)                       # @inspect num_tokens
    return num_bytes / num_tokens


# def get_gpt2_tokenizer():
#     # Code: https://github.com/openai/tiktoken
#     # You can use cl100k_base for the gpt3.5-turbo or gpt4 tokenizer
#     return tiktoken.get_encoding("gpt2")

# def intro_to_tokenization():
#     raise NotImplementedError

# def tokenization_examples():
#     raise NotImplementedError