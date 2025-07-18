# This code is adapted from:
# https://github.com/karpathy/minbpe/blob/master/minbpe/base.py
# Original author: @karpathy
# Modifications: I modified some documents, the code is basically unchanged.
import os
import time
import regex as re
import multiprocessing
import json
from typing import BinaryIO
from typing import Iterable
from dataclasses import dataclass
from collections import Counter, defaultdict

PAT_GPT2 = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
PAT_SPECIAL_TOKEN = {
    '<|endoftext|>': 50256
}


def get_stats(ids, counts=None):
    counts = counts or Counter()
    for pair in zip(ids, ids[1:]):
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
    def __init__(self, vocab, merges, special_tokens=None):
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
        text_chunks = re.finditer(PAT_GPT2, text)
        ids = []
        for chunk in text_chunks:
            chunk_text = chunk.group(0)
            chunk_bytes = chunk_text.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)

        return ids

    
def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk_for_stats(args):
    file_path, start, end = args

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    chunk_text = chunk_bytes.decode("utf-8", errors="ignore")

    text_chunks = PAT_GPT2.finditer(chunk_text)

    chunk_stats = Counter()
    for chunk in text_chunks:
        chunk_str = chunk.group(0)
        chunk_bytes = chunk_str.encode("utf-8")
        indices = list(chunk_bytes)
        get_stats(indices, chunk_stats)

    return chunk_stats


def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_token: dict[str, int],
    num_processes: None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}

    max_merges = vocab_size - 256 - len(special_token)

    with multiprocessing.Pool(processes=num_processes) as pool:
        for merge_step in range(max_merges):

            chunk_args = [(input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
            chunk_results = pool.map(process_chunk_for_stats, chunk_args)

            total_stats = Counter()
            for chunk_stats in chunk_results:
                total_stats.update(chunk_stats)

            if not total_stats:
                print(f"No more pairs to merge after {merge_step} steps")
                break

            best_pair = sorted(total_stats.items(), key=lambda x: (-x[1], x[0]))[0][0]
            p0, p1 = best_pair
            new_index = 256 + merge_step
            vocab[new_index] = vocab[p0] + vocab[p1]
            merges.append((vocab[p0], vocab[p1]))

            if (merge_step + 1) % 10 == 0:
                print(f" Progress: {merge_step + 1}/{max_merges} merges completed")

    for token, token_id in special_token.items():
        vocab[token_id] = token.encode("utf-8")

    return (vocab, merges)


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")