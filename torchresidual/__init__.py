"""
torchresidual
~~~~~~~~~~~~~

Flexible residual connections for PyTorch.

    from torchresidual import ResidualSequential, Record, Apply, LearnableAlpha
"""

from .alpha import LearnableAlpha
from .wrappers import RecurrentWrapper
from .core import Record, Apply, ResidualSequential

__version__ = "0.1.0"
__all__ = [
    "LearnableAlpha",
    "RecurrentWrapper",
    "Record",
    "Apply",
    "ResidualSequential",
]
