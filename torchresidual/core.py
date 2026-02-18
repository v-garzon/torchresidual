import threading
import warnings
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .alpha import LearnableAlpha
from .wrappers import RecurrentWrapper

# Thread-local storage: avoids circular parent references while remaining
# safe for use with nn.DataParallel (each thread gets its own slot).
_context = threading.local()


class Record(nn.Module):
    """
    Records the current tensor at this point in the network for later use
    in a residual connection via :class:`Apply`.

    Args:
        need_projection: When ``True``, :class:`Apply` will automatically
            create a linear projection if the recorded shape differs from the
            current tensor shape.
        name: Optional label used to reference this record point from
            :class:`Apply`.  Auto-assigned if not provided.

    Example::

        block = ResidualSequential(
            Record(name="input"),
            nn.Linear(64, 64),
            Apply(record_name="input"),
        )
    """

    def __init__(self, need_projection: bool = False, name: Optional[str] = None):
        super().__init__()
        self.need_projection = need_projection
        self.name = name
        self.recorded_tensor: Optional[torch.Tensor] = None
        self.recorded_shape: Optional[torch.Size] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.recorded_tensor = x
        self.recorded_shape = x.shape
        return x

    def __repr__(self) -> str:
        return f"Record(need_projection={self.need_projection}, name={self.name!r})"


class Apply(nn.Module):
    """
    Applies a residual connection using a tensor previously saved by
    :class:`Record`.  Must be used inside :class:`ResidualSequential`.

    Args:
        operation: One of ``"add"``, ``"concat"``, ``"multiply"``,
            ``"gated"``, ``"highway"``.  Default is ``"add"``.
        record_name: Name of the :class:`Record` to retrieve.  If ``None``,
            the most-recently encountered :class:`Record` is used.
        alpha: Scalar weight applied to the residual branch.  Accepts a
            plain ``float`` or a :class:`~torchresidual.LearnableAlpha`.

    Operations:

    * ``add``      — ``x + alpha * r`` (default)
    * ``concat``   — ``torch.cat([x, r], dim=-1)``
    * ``multiply`` — ``x * (1 + alpha * r)``
    * ``gated``    — ``(1 - alpha) * x + alpha * r``
    * ``highway``  — dual learnable gates (transform + carry)

    Example::

        Apply(operation="add", record_name="input", alpha=0.5)
        Apply(operation="gated", alpha=LearnableAlpha(0.3))
    """

    _VALID_OPS = {"add", "concat", "multiply", "gated", "highway"}

    def __init__(
        self,
        operation: str = "add",
        record_name: Optional[str] = None,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.operation = operation.lower()
        self.record_name = record_name
        self.alpha = alpha
        self.projection_layer: Optional[nn.Module] = None
        self.gate_layer: Optional[nn.ModuleDict] = None

        if self.operation not in self._VALID_OPS:
            raise ValueError(
                f"Unsupported operation {self.operation!r}. "
                f"Choose from {sorted(self._VALID_OPS)}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parent: Optional["ResidualSequential"] = getattr(
            _context, "current_sequential", None
        )
        if parent is None:
            raise RuntimeError(
                "Apply must be used inside ResidualSequential. "
                "Direct calls to Apply.forward() are not supported."
            )

        recorded_tensor, record_module = parent.get_recorded_tensor(self.record_name)

        if recorded_tensor is None or record_module is None:
            raise RuntimeError(
                f"No recorded tensor found for name={self.record_name!r}. "
                "Make sure a Record module appears before this Apply."
            )

        # ---- shape mismatch handling ----------------------------------------
        if recorded_tensor.shape != x.shape:
            if record_module.need_projection:
                if self.projection_layer is None:
                    self.projection_layer = self._create_projection_layer(
                        recorded_tensor.shape, x.shape
                    ).to(x.device)
                recorded_tensor = self.projection_layer(recorded_tensor)
            else:
                try:
                    if self.operation in {"add", "gated", "highway", "multiply"}:
                        torch.broadcast_shapes(recorded_tensor.shape, x.shape)  # type: ignore[arg-type]
                    elif self.operation == "concat":
                        if recorded_tensor.shape[:-1] != x.shape[:-1]:
                            raise RuntimeError("Batch dims must match for concat.")
                except RuntimeError:
                    raise RuntimeError(
                        f"Shape mismatch: recorded {recorded_tensor.shape} vs "
                        f"current {x.shape}. Set need_projection=True in Record() "
                        "to enable automatic projection."
                    )

        # ---- apply operation ------------------------------------------------
        alpha = self.alpha() if isinstance(self.alpha, LearnableAlpha) else self.alpha

        if self.operation == "add":
            return x + alpha * recorded_tensor
        if self.operation == "concat":
            return torch.cat([x, recorded_tensor], dim=-1)
        if self.operation == "multiply":
            return x * (1 + alpha * recorded_tensor)
        if self.operation == "gated":
            return (1 - alpha) * x + alpha * recorded_tensor
        if self.operation == "highway":
            if self.gate_layer is None:
                dim = x.shape[-1]
                self.gate_layer = nn.ModuleDict(
                    {
                        "transform_gate": nn.Linear(dim, dim),
                        "carry_gate": nn.Linear(dim, dim),
                    }
                ).to(x.device)
                self.gate_layer["transform_gate"].bias.data.fill_(-1.0)  # type: ignore[operator]
                self.gate_layer["carry_gate"].bias.data.fill_(1.0)  # type: ignore[operator]
            t = torch.sigmoid(self.gate_layer["transform_gate"](x))
            c = torch.sigmoid(self.gate_layer["carry_gate"](recorded_tensor))
            return t * x + c * recorded_tensor

        raise RuntimeError(f"Unreachable operation: {self.operation!r}")  # pragma: no cover

    # ------------------------------------------------------------------ helpers
    def _create_projection_layer(
        self, from_shape: torch.Size, to_shape: torch.Size
    ) -> nn.Module:
        if len(from_shape) != len(to_shape):
            raise RuntimeError(
                "Cannot project between tensors with different number of dimensions."
            )
        if from_shape[-1] != to_shape[-1]:
            return nn.Linear(from_shape[-1], to_shape[-1])
        return nn.Identity()

    def __repr__(self) -> str:
        return (
            f"Apply(operation={self.operation!r}, "
            f"record_name={self.record_name!r}, "
            f"alpha={self.alpha})"
        )


class ResidualSequential(nn.Sequential):
    """
    A drop-in replacement for ``nn.Sequential`` that supports flexible
    residual connections via :class:`Record` and :class:`Apply` modules.

    Uses thread-local storage to pass context to :class:`Apply` without
    circular parent references, making it safe for ``nn.DataParallel`` and
    multi-threaded inference servers.

    Features:

    * Multiple named record points
    * Five residual operation types (``add``, ``concat``, ``multiply``,
      ``gated``, ``highway``)
    * Automatic projection for shape mismatches (``need_projection=True``)
    * Learnable alpha via :class:`~torchresidual.LearnableAlpha`
    * Recurrent module support via :class:`~torchresidual.RecurrentWrapper`
    * Device-agnostic; works on CPU and CUDA

    Limitations:

    * **TorchScript / ONNX export not supported** (planned for v1.1)
    * ``nn.DataParallel`` ✅ ``nn.parallel.DistributedDataParallel`` ✅

    Example::

        block = ResidualSequential(
            Record(name="input"),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            Apply(record_name="input", operation="add"),
        )
        out = block(x)
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.recorded_tensors: Dict[str, Tuple[torch.Tensor, Record]] = {}
        self.last_record_name: Optional[str] = None
        self.extra_output: list = []
        self._setup_modules()

    def _setup_modules(self) -> None:
        record_count = 0
        apply_count = 0
        for module in self:
            if isinstance(module, Apply):
                apply_count += 1
            elif isinstance(module, Record):
                record_count += 1
                if module.name is None:
                    module.name = f"record_{record_count}"
        if apply_count > 0 and record_count == 0:
            warnings.warn(
                "ResidualSequential contains Apply modules but no Record modules. "
                "This will raise a RuntimeError at forward time.",
                stacklevel=2,
            )

    def get_recorded_tensor(
        self, name: Optional[str] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[Record]]:
        """Return ``(tensor, record_module)`` for *name* or the most recent record."""
        key = name if name is not None else self.last_record_name
        if key is None or key not in self.recorded_tensors:
            return None, None
        tensor, module = self.recorded_tensors[key]
        return tensor, module

    @contextmanager
    def _execution_context(self):
        """Set this instance as the active context for Apply modules."""
        old = getattr(_context, "current_sequential", None)
        _context.current_sequential = self
        try:
            yield
        finally:
            _context.current_sequential = old

    def forward(self, x: torch.Tensor, hidden=None, **kwargs) -> torch.Tensor:  # type: ignore[override]
        """
        Args:
            x: Input tensor.
            hidden: Optional hidden state forwarded to :class:`RecurrentWrapper`.
            **kwargs: Per-module extra keyword arguments, keyed by the module's
                name in the ``OrderedDict`` (only when constructed with one).

        Returns:
            Output tensor, or ``(output, extra_outputs)`` when any sub-module
            returns a tuple.
        """
        self.extra_output = []

        with self._execution_context():
            for name, module in self.named_children():
                extra = kwargs.get(name, {})

                if isinstance(module, RecurrentWrapper):
                    x = module(x, hidden) if hidden is not None else module(x)
                elif isinstance(module, Record):
                    x = module(x)
                    assert module.recorded_tensor is not None  # Record always stores tensor
                    self.recorded_tensors[module.name] = (module.recorded_tensor, module)  # type: ignore[index]
                    self.last_record_name = module.name
                else:
                    x = module(x, **extra) if extra else module(x)

                if isinstance(x, tuple):
                    self.extra_output.extend(x[1:])
                    x = x[0]

        if self.extra_output:
            return x, self.extra_output  # type: ignore[return-value]
        return x

    def get_residual_info(self) -> dict:
        """Return a summary of Record/Apply modules and their connections."""
        info: dict = {"records": [], "applies": []}
        for module in self:
            if isinstance(module, Record):
                info["records"].append(
                    {
                        "name": module.name,
                        "need_projection": module.need_projection,
                        "shape": module.recorded_shape,
                    }
                )
            elif isinstance(module, Apply):
                info["applies"].append(
                    {
                        "operation": module.operation,
                        "record_name": module.record_name,
                        "alpha": module.alpha,
                    }
                )
        return info

    def __repr__(self, depth: int = 1) -> str:
        lines = [f"{self.__class__.__name__}("]
        indent = "  "
        for i, module in enumerate(self):
            if isinstance(module, ResidualSequential):
                module_repr = module.__repr__(depth=depth + 1)  # type: ignore[call-arg]
                lines.append(f"{indent * depth}({i}): {module_repr}")
            else:
                lines.append(f"{indent * depth}({i}): {module}")
        lines.append(f"{indent * max(depth - 1, 0)})")
        return "\n".join(lines)
