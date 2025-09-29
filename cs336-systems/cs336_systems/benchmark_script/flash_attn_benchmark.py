import timeit
import itertools
from typing import Callable
import torch
from cs336_basics.modules import MultiHeadSelfAttention
from cs336_basics.modules import AdamW
from profiling_benchmark import *
from flash_attention import *


def flash_benchmarking():
    raise NotImplementedError