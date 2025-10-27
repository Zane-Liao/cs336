"""
Come from CS336 lecture14 Some Example Code...
Author: Percy Liang
"""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import numpy as np
import kenlm
import fasttext
import itertools
import mmh3
from bitarray import bitarray


def compute(content: str, model: kenlm.Model):
    # Hacky preprocessing
    content = "<s> " + content.replace(",", " ,").replace(".", " .") + " </s>"
    # log p(content)
    score = model.score(content)
    # Perplexity normalizes by number of tokens to avoid favoring short documents
    num_tokens = len(list(model.full_scores(content)))
    perplexity = math.exp(-score / num_tokens)
    
    return score, perplexity


def importance_sampling():
    vocabulary = [0, 1, 2, 3]
    p = [0.1, 0.2, 0.3, 0.4]
    q = [0.4, 0.3, 0.2, 0.1]

    # 1. Sample from q
    n = 100
    samples = np.random.choice(vocabulary, p=q, size = n)  # @inspect samples
    # Samples (q): [0 2 0 2 1 1 0 1 0 1 0 0 0 0 1 1 0 3 1 1 1 1 1 1 3 1 0 2 0 1 3 0 2 1 1 2 0 0 0 3
    # 2 1 1 0 1 1 0 3 2 0 2 0 1 0 1 2 2 2 0 0 0 0 2 1 0 2 0 1 3 0 0 0 0 0 0 0 1 1 2 2 2 1 0 1 1 0 1
    # 0 1 0 1 1 3 3 0 1 0 0 2 0]
    
    # 2. Compute weights over samples (w \propto p/q)
    w = [p[x] / q[x] for x in samples]  # @inspect w
    z = sum(w)  # @inspect z
    w = [w_i / z for w_i in w]  # @inspect w

    # 3. Resample
    samples = np.random.choice(samples, p=w, size=n)  # @inspect samples
    # Resampled (p): [2 2 1 3 3 2 0 3 3 2 1 0 3 2 0 3 3 1 3 1 3 2 3 2 3 2 3 1 0 3 2 2 2 0 2 1 2 0 3
    # 1 1 1 3 1 3 3 3 1 0 2 3 1 2 1 2 2 2 2 1 1 0 2 1 1 0 2 3 2 1 3 2 3 1 2 3 2 3 2 1 1 3 3 3 1 1 3
    # 2 1 3 1 3 1 0 3 1 1 2 3 2 1]


def bloom_filter():
    # Goal: efficient, approximate data structure for testing set membership
    # Features of Bloom filters
    # - Memory efficient
    # - Can update, but can't delete
    # - If return 'no', definitely 'no'
    # - If return 'yes', most likely 'yes', but small probability of 'no'
    
    # Can drive the false positive rate down exponentially with more time/compute
    items = ["the", "cat", "in", "the", "hat"]
    non_items = ["what", "who", "why", "when", "where", "which", "how"]

    # First, make the range of hash function small (small number of bins).
    m = 8  # Number of bins
    table = build_table(items, m)
    for item in items:
        assert query_table(table, item, m) == 1
    result = {item: query_table(table, item, m) for item in non_items}  # @inspect result
    num_mistakes = count(result.values(), True)  # @inspect num_mistakes
    false_positive_rate = num_mistakes / (len(items) + num_mistakes)  # @inspect false_positive_rate

    # Problem: false positives for small bins
    # Naive solution: increase the number of bins
    # Error probability is O(1/num_bins), decreases polynomially with memory
    # Better solution: use more hash functions
    k = 2  # Number of hash functions
    table = build_table_k(items, m, k)
    for item in items:
        assert query_table_k(table, item, m, k) == 1
    result = {item: query_table_k(table, item, m, k) for item in non_items}  # @inspect result
    num_mistakes = count(result.values(), 1)  # @inspect num_mistakes
    false_positive_rate = num_mistakes / (len(items) + num_mistakes)  # @inspect false_positive_rate

    # Reduced the false positive rate!
    false_positive_rate_analysis()
    

def false_positive_rate_analysis():
    # Assume independence of hash functions and items  [article]
    m = 1000   # Number of bins
    k = 10     # Number of hash functions
    n = 100    # Number of items we're inserting
    # Consider a test input (not in the set) that would hash into a given test bin (say, i).
    # Now consider putting items into the Bloom filter and seeing if it hits i.
    # Insert one item, ask if the test bin B(i) = 1?
    # B: [0 0 1 0 0 0 0 0 0 0] - have to miss 1 time
    f = 1 / m                              # P[B(i) = 1 after 1 insertion with 1 hash function]  # @inspect f
    # B: [0 0 1 0 0 1 0 1 0 0] - have to miss k times
    f = 1 - (1 - 1 / m) ** k               # P[B(i) = 1 after 1 insertion with k hash functions]  # @inspect f
    # Insert n items, ask if the test bin B(i) = 1?
    # Have to miss k*n times
    f = 1 - (1 - 1 / m) ** (k * n)         # P[B(i) = 1 after n insertions for 1 hash function]  # @inspect f
    # Get k chances to miss (since test input is hashed k times too)
    f = f ** k                             # P[B(i) = 1 after n insertions for k hash functions]  # @inspect f
    # Optimal value of k (given fixed m / n ratio) [results in f ~ 0.5]
    k = math.log(2) * m / n  # @inspect k
    # Resulting false positive rate (improved)
    f = 0.5 ** k  # @inspect f
    # Tradeoff between compute (k), memory (m), and false positive rate (f)  [lecture notes]
    # Example: Dolma
    # - Set false positive rate to 1e-15
    # - Perform on items = paragraphs

    
##################################################################################################
##################################################################################################


def build_table(items: list[str], num_bins: int):
    """Build a Bloom filter table of size `num_bins`, inserting `items` into it."""
    table = bitarray(num_bins)  # @inspect table
    for item in items:
        h = mmh3.hash(item) % num_bins  # @inspect item, @inspect h
        table[h] = 1  # @inspect table
    return table


def build_table_k(items: list[str], num_bins: int, k: int):
    """Build a Bloom filter table of size `num_bins`, inserting `items` into it.
    Use `k` hash functions."""
    table = bitarray(num_bins)  # @inspect table
    for item in items:
        # For each of the k functions
        for seed in range(k):
            h = mmh3.hash(item, seed) % num_bins  # @inspect item, @inspect h, @inspect seed
            table[h] = 1  # @inspect table
    return table


def query_table(table: bitarray, item: str, num_bins: int, seed: int = 0):
    """Return whether `item` is in the `table`."""
    h = mmh3.hash(item, seed) % num_bins
    return table[h]


def query_table_k(table: bitarray, item: str, num_bins: int, k: int):
    """Return 1 if table set to 1 for all `k` hash functions."""
    return int(all(
        query_table(table, item, num_bins, seed)
        for seed in range(k)
    ))
    
    
def compute_jaccard(A, B):
        intersection = len(A & B)  # @inspect intersection
        union = len(A | B)  # @inspect union
        return intersection / union
    
    
def minhash(S: set[str], seed: int):
    return min(mmh3.hash(x, seed) for x in S)


def get_prob_collision(sim, b, r):  # @inspect sim, @inspect b, @inspect r
    prob_match = sim ** r                        # Probability that a fixed band matches  @inspect prob_match
    prob_collision = 1 - (1 - prob_match) ** b   # Probability that some band matches  @inspect prob_collision
    
    return prob_collision

##################################################################################################
##################################################################################################

def round1(x: float) -> float:
    """Round to 1 decimal place."""
    return round(x, 1)


def mean(x: list[float]) -> float:
    return sum(x) / len(x)


def count(list, x):
    """Return the number of times `x` appears in `list`."""
    return sum(1 for y in list if y == x)


def repeat(f, n: int):
    """Return a list with the results of calling function `f` `n` times."""
    return [f() for _ in range(n)]