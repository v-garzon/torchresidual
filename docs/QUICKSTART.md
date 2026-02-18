# Quick Start Guide

This guide will get you up and running with `torchresidual` in 5 minutes.

## Installation

```bash
pip install torchresidual
```

**Requirements:**
- Python ≥ 3.8
- PyTorch ≥ 1.9

## Your First Residual Block

```python
import torch
import torch.nn as nn
from torchresidual import ResidualSequential, Record, Apply

# Create a simple residual block
block = ResidualSequential(
    Record(name="input"),          # Save tensor here
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.LayerNorm(64),
    Apply(record_name="input"),    # Add saved tensor here
)

# Use it like any nn.Module
x = torch.randn(8, 64)
output = block(x)
print(output.shape)  # torch.Size([8, 64])
```

**What happened?**
1. `Record` saves the input tensor
2. Layers transform the tensor
3. `Apply` adds the saved tensor back: `output = transformed + saved`

## Common Patterns

### Pattern 1: Change dimensions with projection

```python
# Input: [batch, 64] → Output: [batch, 128]
block = ResidualSequential(
    Record(name="r", need_projection=True),  # Enable auto-projection
    nn.Linear(64, 128),
    nn.ReLU(),
    Apply(record_name="r"),  # Automatically projects 64→128
)
```

### Pattern 2: Multiple skip connections

```python
block = ResidualSequential(
    Record(name="input", need_projection=True),
    nn.Linear(64, 32),
    Record(name="mid"),
    nn.Linear(32, 64),
    Apply(record_name="input"),  # Long skip
    nn.Linear(64, 32),
    Apply(record_name="mid"),     # Short skip
)
```

### Pattern 3: Learnable mixing coefficient

```python
from torchresidual import LearnableAlpha

block = ResidualSequential(
    Record(name="r"),
    nn.Linear(64, 64),
    Apply(
        record_name="r",
        operation="gated",  # (1-α)·x + α·residual
        alpha=LearnableAlpha(0.5, min_value=0.0, max_value=1.0),
    ),
)

# Alpha learns during training!
optimizer = torch.optim.Adam(block.parameters(), lr=1e-3)
```

### Pattern 4: LSTM with residual

```python
from torchresidual import RecurrentWrapper

block = ResidualSequential(
    Record(name="r"),
    RecurrentWrapper(
        nn.LSTM(32, 32, num_layers=2, batch_first=True),
        return_hidden=False,
    ),
    Apply(record_name="r"),
)

x = torch.randn(4, 10, 32)  # [batch, seq, features]
output = block(x)
```

## Operations

Five residual operations are available:

```python
# Standard addition (ResNet-style)
Apply(operation="add")  # x + α·r

# Concatenation (DenseNet-style)
Apply(operation="concat")  # cat([x, r], dim=-1)

# Multiplication
Apply(operation="multiply")  # x·(1 + α·r)

# Gated (learnable interpolation)
Apply(operation="gated")  # (1-α)·x + α·r

# Highway network (dual gates)
Apply(operation="highway")  # T·x + C·r
```

## Complete Example

```python
import torch
import torch.nn as nn
from torchresidual import ResidualSequential, Record, Apply, LearnableAlpha

# Build a residual encoder
encoder = ResidualSequential(
    # First block: 64 → 128 with projection
    Record(name="block1", need_projection=True),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.LayerNorm(128),
    Apply(record_name="block1"),
    
    # Second block: 128 → 128, no projection needed
    Record(name="block2"),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.LayerNorm(128),
    Apply(record_name="block2", alpha=LearnableAlpha(0.5)),
)

# Training loop
optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

for epoch in range(10):
    x = torch.randn(32, 64)  # Batch of 32
    target = torch.randn(32, 128)
    
    output = encoder(x)
    loss = ((output - target) ** 2).mean()
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")
```

## Next Steps

- Check out [examples/](../examples/) for more complex patterns
- Read the [full documentation](../README.md) for API details
- See [DESIGN.md](DESIGN.md) for implementation details

## Need Help?

- **Documentation:** See [README.md](../README.md)
- **Examples:** Browse [examples/](../examples/)
- **Issues:** https://github.com/v-garzon/torchresidual/issues
