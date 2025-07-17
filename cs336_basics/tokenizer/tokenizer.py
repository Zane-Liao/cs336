# This code is adapted from:
# https://github.com/karpathy/minbpe/blob/master/minbpe/base.py
# Original author: @karpathy
# Modifications: I modified some documents, the code is basically unchanged.

import regex as re
import random
import unicodedata
import doctest
import multiprocessing
import json
from typing import Iterable

PAT_GPT2 = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_SPECIAL_TOKEN = {
    '<|endoftext|>': 50256
}


# @karpathy
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

# @karpathy
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


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=PAT_SPECIAL_TOKEN):
        """
        Construct a tokenizer from a given vocabulary, list of merges,
        and (optionally) a list of special tokens.

        Parameters:
            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
            pecial_tokens: list[str] | None = None
        """
        self.merges = merges
        self.special_tokens = special_tokens or {}
        self.vocab = self._build_vocab()
    
    def encode(self, string: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        
        Parameters:
            string: str
        Return:
            list[int]
        """
        if self.special_tokens is None:
            self.register_special_tokens({"<|endoftext|>": 50256})

        return self.encode_ordinary(string)
        
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Class method that constructs and return a Tokenizer from a
        serialized vocabulary and list of merges (in the same format
        that your BPE training code output) and (optionally) a list of special
        tokens.
        
        Parameters:
            cls: Class Tokenizer
            vocab_filepath: str
            merges_filepath: str
            special_tokens: list[str] | None = None
        """
        with open (vocab_filepath, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        with open (merges_filepath, 'r', encoding='utf-8') as f:
            merges = [line.strip().split() for line in f if not line.startswith("#")]
            
        return cls(vocab, merges, special_tokens=special_tokens)
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        Given an iterable of strings (e.g., a Python file handle),
        return a generator that lazily yields token IDs. This is
        required for memory-eﬀicient tokenization of large files that
        we cannot directly load into memory.
        
        Parameter:
            iterable: Iterable[str]
        Return:
            Iterable[int]
        """
        for string in iterable:
            string = string.strip("\n")
            token = self.encode(string)
            for token_id in token:
                yield token_id
    
    def decode(self, indices: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        
        Parameter:
            indices: list[int]
        Return:
            str
        """
        # @karpathy
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in indices:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        string = text_bytes.decode("utf-8", errors="replace")
        return string
    
    # @karpathy
    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
    
    # @karpathy
    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 50256}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    # @karpathy
    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
    
    # @karpathy
    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.finditer(PAT_GPT2, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

def train_bpe(
    input_path: str, 
    vocab_size: int, 
    speical_token: list[str],
) -> tuple[dict[int, bytes], dict[tuple[int, int], int]]:
    """
    input_path: str Path to a text file with BPE tokenizer training data.
    
    vocab_size: int A positive integer that defines the maximum final vocabulary
    size (including the initial byte vocabulary, vocabulary items produced 
    from merging, and any special tokens).
    
    special_tokens: list[str] A list of strings to add to the vocabulary.
    These special tokens do not otherwise affect BPE training.
    
    vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int
    (token ID in the vocabu-lary) to bytes (token bytes).
    
    merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training.
    Each list item is a tuple of bytes (<token1>, <token2>),
    representing that <token1> was merged with <token2>.
    The merges should be ordered by order of creation.
    """
    raise NotImplementedError