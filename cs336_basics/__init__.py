import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .modules import (
    transformer,
    train_transformer,
)

from .tokenizer import (
    tokenizer,
    train_tokenizer,
)