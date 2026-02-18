import math
import torch
import torch.nn as nn
from typing import Optional


class LearnableAlpha(nn.Module):
    """
    A learnable scalar parameter constrained to [min_value, max_value].

    Uses a tanh-based reparameterization to enforce bounds during training.
    Supports linear and log space; log space is auto-selected when the range
    spans more than two orders of magnitude (and min_value > 0).

    Args:
        initial_value: Starting value for the parameter.
        min_value: Lower bound (inclusive).
        max_value: Upper bound (inclusive).
        use_log_space: Force log (True) or linear (False) space. If None,
            auto-detects from the min/max ratio.

    Example::

        alpha = LearnableAlpha(0.5, min_value=0.0, max_value=1.0)
        out = x + alpha() * residual
    """

    def __init__(
        self,
        initial_value: float = 1.0,
        min_value: float = 0.0,
        max_value: float = 1.0,
        use_log_space: Optional[bool] = None,
    ):
        super().__init__()

        if min_value >= max_value:
            raise ValueError(
                f"min_value ({min_value}) must be strictly less than max_value ({max_value})"
            )

        self.register_buffer("min_val", torch.tensor(min_value, dtype=torch.float32))
        self.register_buffer("max_val", torch.tensor(max_value, dtype=torch.float32))

        if use_log_space is None:
            ratio = max_value / min_value if min_value > 0 else float("inf")
            self.use_log_space: bool = ratio > 100 and min_value > 0
        else:
            self.use_log_space = use_log_space

        self.register_buffer("use_log", torch.tensor(self.use_log_space))

        init_logit = (
            self._get_logit_initial_value_log(initial_value)
            if self.use_log_space
            else self._get_logit_initial_value_linear(initial_value)
        )
        self.param = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

    # ------------------------------------------------------------------ init helpers
    def _get_logit_initial_value_linear(self, value: float) -> float:
        lo, hi = self.min_val.item(), self.max_val.item()  # type: ignore[union-attr,operator]
        value = max(lo + 1e-6, min(hi - 1e-6, value))
        normalized = (2 * value - lo - hi) / (hi - lo)
        return math.atanh(max(-0.999, min(0.999, normalized)))

    def _get_logit_initial_value_log(self, value: float) -> float:
        lo, hi = self.min_val.item(), self.max_val.item()  # type: ignore[union-attr,operator]
        value = max(lo * 1.001, min(hi * 0.999, value))
        log_min, log_max = math.log(lo), math.log(hi)
        normalized = (2 * math.log(value) - log_min - log_max) / (log_max - log_min)
        return math.atanh(max(-0.999, min(0.999, normalized)))

    # ------------------------------------------------------------------ forward
    def _get_value(self) -> torch.Tensor:
        """Return the constrained parameter value."""
        t = torch.tanh(self.param)
        if self.use_log_space:
            log_min = torch.log(self.min_val)  # type: ignore[arg-type]
            log_max = torch.log(self.max_val)  # type: ignore[arg-type]
            log_val = (log_min + log_max) / 2 + (log_max - log_min) / 2 * t  # type: ignore[operator]
            return torch.exp(log_val)
        return (self.min_val + self.max_val) / 2 + (self.max_val - self.min_val) / 2 * t  # type: ignore[operator,return-value]

    # ------------------------------------------------------------------ operators
    def __mul__(self, other):       return self._get_value() * other
    def __rmul__(self, other):      return other * self._get_value()
    def __add__(self, other):       return self._get_value() + other
    def __radd__(self, other):      return other + self._get_value()
    def __sub__(self, other):       return self._get_value() - other
    def __rsub__(self, other):      return other - self._get_value()
    def __truediv__(self, other):   return self._get_value() / other
    def __rtruediv__(self, other):  return other / self._get_value()
    def __pow__(self, other):       return self._get_value() ** other
    def __rpow__(self, other):      return other ** self._get_value()

    def __call__(self) -> torch.Tensor:  # type: ignore[override]
        return self._get_value()

    def __repr__(self) -> str:
        space = "log" if self.use_log_space else "linear"
        min_v = self.min_val.item() if isinstance(self.min_val, torch.Tensor) else self.min_val  # type: ignore[union-attr]
        max_v = self.max_val.item() if isinstance(self.max_val, torch.Tensor) else self.max_val  # type: ignore[union-attr]
        return (
            f"LearnableAlpha(raw={self.param.item():.4f}, "
            f"value={self._get_value().item():.4f}, "
            f"space={space}, "
            f"bounds=[{min_v}, {max_v}])"
        )
