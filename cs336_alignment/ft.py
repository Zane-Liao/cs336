"""
Fine-Tuning
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer


class Dataload(Dataset):
    """
    Test1:
        implement the test adapter at [adapters.get_packed_sft_dataset]
        uv run pytest -k test_ packed_sft_dataset
    Test2:
        implement the test adapter at [adapters.run_iterate_batches]
        uv run pytest -k test_iterate_ batches
    """
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, i):
        raise NotImplementedError


def sft_script():
    raise NotImplementedError

def mmlu_sft():
    raise NotImplementedError

def gsm8k_sft():
    raise NotImplementedError

def alpaca_eval_sft():
    raise NotImplementedError

def sst_sft():
    raise NotImplementedError
