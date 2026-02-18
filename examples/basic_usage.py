"""Basic usage examples for torchresidual."""
import torch
import torch.nn as nn
from torchresidual import ResidualSequential, Record, Apply, LearnableAlpha

# ── Example 1: simple residual add ───────────────────────────────────────────
block = ResidualSequential(
    Record(name="input"),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.LayerNorm(64),
    Apply(record_name="input", operation="add"),
)

x = torch.randn(8, 64)
out = block(x)
print(f"[add] input={x.shape}  output={out.shape}")

# ── Example 2: gated residual with learnable alpha ────────────────────────────
block_gated = ResidualSequential(
    Record(name="r"),
    nn.Linear(64, 64),
    nn.ReLU(),
    Apply(record_name="r", operation="gated", alpha=LearnableAlpha(0.3, 0.0, 1.0)),
)

out2 = block_gated(x)
print(f"[gated] input={x.shape}  output={out2.shape}")

# ── Example 3: projection when dims differ ────────────────────────────────────
block_proj = ResidualSequential(
    Record(name="r", need_projection=True),
    nn.Linear(64, 128),
    nn.ReLU(),
    Apply(record_name="r", operation="add"),
)

out3 = block_proj(x)
print(f"[projection] input={x.shape}  output={out3.shape}")
