""""""
from dataclasses import dataclass
from typing import Callable, List, Literal
import torch
import torch.nn as nn
import wandb


def mmlu_baseline():
    """
    Test:
        implement the adapter [run_parse_mmlu_response]
        uv run pytest -k test_parse_mmlu_response
    """
    raise NotImplementedError


def gsm8k_baseline():
    """
    Test:
        implement the adapter [run_parse_gsm8k_response]
        uv run pytest -k test_parse_gsm8k_ response
    """
    raise NotImplementedError
    

def alpaca_eval_baseline():
    raise NotImplementedError


def sst_baseline():
    raise NotImplementedError


