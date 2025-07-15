import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .model import Model
from .transformer import DenseTransformer