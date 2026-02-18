import torch
import torch.nn as nn
from typing import Optional, Tuple


class RecurrentWrapper(nn.Module):
    """
    Thin wrapper around any PyTorch recurrent module (LSTM, GRU, …) that
    normalises the ``(output, hidden)`` API so it integrates cleanly with
    :class:`ResidualSequential`.

    Args:
        module: The underlying recurrent module (e.g. ``nn.LSTM(…)``).
        return_hidden: If ``True`` the wrapper returns ``(output, hidden)``
            as a tuple; if ``False`` only ``output`` is returned.

    Example::

        block = ResidualSequential(
            Record(need_projection=False),
            RecurrentWrapper(nn.LSTM(64, 64), return_hidden=False),
            Apply(),
        )
        out = block(x)
    """

    def __init__(self, module: nn.Module, return_hidden: bool = False):
        super().__init__()
        self.module = module
        self.return_hidden = return_hidden

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if hidden is not None:
            output, h_n = self.module(x, hidden)
        else:
            output, h_n = self.module(x)

        if self.return_hidden:
            return output, h_n
        return output

    def __repr__(self) -> str:
        return f"RecurrentWrapper(return_hidden={self.return_hidden}, module={self.module})"
