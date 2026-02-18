# torchresidual

[![PyPI version](https://badge.fury.io/py/torchresidual.svg)](https://badge.fury.io/py/torchresidual)
[![Python versions](https://img.shields.io/pypi/pyversions/torchresidual.svg)](https://pypi.org/project/torchresidual/)
[![Tests](https://github.com/v-garzon/torchresidual/actions/workflows/tests.yml/badge.svg)](https://github.com/v-garzon/torchresidual/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/v-garzon/torchresidual/blob/main/LICENSE)

**Flexible residual connections for PyTorch with a clean, composable API.**

Build complex residual architectures without boilerplate. `torchresidual` provides
`Record` and `Apply` modules that let you create skip connections of any depth,
with automatic shape handling and learnable mixing coefficients.

---

> 📖 **[Quick Start](docs/QUICKSTART.md)** |
> 📚 **[Full Documentation](#api-reference)** |
> 💡 **[Examples](examples/)** |
> ❓ **[FAQ](docs/FAQ.md)**

---

## Why torchresidual?

**Standard PyTorch:**

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
  
    def forward(self, x):
        residual = x
        x = self.linear(x)
        x = F.relu(x)
        x = self.norm(x)
        return x + residual  # Manual residual
```

**With torchresidual:**

```python
block = ResidualSequential(
    Record(name="input"),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.LayerNorm(64),
    Apply(record_name="input"),  # Automatic residual
)
```

**Benefits:**

- No custom `forward()` methods
- Multiple skip connections with named records
- Automatic projection when dimensions change
- Five residual operations (add, concat, multiply, gated, highway)
- Learnable mixing coefficients
- Works with LSTMs, attention, and any `nn.Module`

---

## Installation

```bash
pip install torchresidual
```

**Requirements:** Python ≥3.9, PyTorch ≥1.9

**New to torchresidual?** See the [Quick Start Guide](docs/QUICKSTART.md) for a 5-minute tutorial.

---

## Quick Start

### Basic residual connection

```python
import torch
import torch.nn as nn
from torchresidual import ResidualSequential, Record, Apply

block = ResidualSequential(
    Record(name="input"),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.LayerNorm(64),
    Apply(record_name="input", operation="add"),
)

x = torch.randn(8, 64)
out = block(x)  # Shape: [8, 64]
```

### Multiple skip connections

```python
block = ResidualSequential(
    Record(name="input", need_projection=True),
    nn.Linear(64, 32),
    nn.ReLU(),
    Record(name="mid"),
    nn.Linear(32, 64),
    Apply(record_name="input"),      # Long skip with projection
    nn.LayerNorm(64),
    nn.Linear(64, 32),
    Apply(record_name="mid"),         # Short skip
)
```

### Learnable mixing coefficient

```python
from torchresidual import LearnableAlpha

block = ResidualSequential(
    Record(name="r"),
    nn.Linear(64, 64),
    Apply(
        record_name="r", 
        operation="gated",
        alpha=LearnableAlpha(0.3, min_value=0.0, max_value=1.0)
    ),
)

# Alpha is learned during training
optimizer = torch.optim.Adam(block.parameters(), lr=1e-3)
```

### Automatic projection for shape changes

```python
# Input: [batch, 64] → Output: [batch, 128]
block = ResidualSequential(
    Record(name="r", need_projection=True),  # Enables auto-projection
    nn.Linear(64, 128),
    nn.ReLU(),
    Apply(record_name="r"),  # Automatically projects 64→128
)
```

### LSTM with residual

```python
from torchresidual import RecurrentWrapper

block = ResidualSequential(
    Record(name="r"),
    RecurrentWrapper(
        nn.LSTM(32, 32, num_layers=2, batch_first=True),
        return_hidden=False
    ),
    Apply(record_name="r"),
)

x = torch.randn(4, 10, 32)  # [batch, seq_len, features]
out = block(x)
```

---

## API Reference

### Core Components

#### `ResidualSequential(*modules)`

Drop-in replacement for `nn.Sequential` with residual connection support.

**Example:**

```python
block = ResidualSequential(
    nn.Linear(64, 64),
    Record(),
    nn.ReLU(),
    Apply(),
)
```

#### `Record(need_projection=False, name=None)`

Saves the current tensor for later use in a residual connection.

**Args:**

- `need_projection` (bool): If `True`, `Apply` will create a linear projection when shapes don't match
- `name` (str, optional): Label for this record point. Auto-assigned if `None`.

**Example:**

```python
Record(name="input", need_projection=True)
```

#### `Apply(operation="add", record_name=None, alpha=1.0)`

Applies a residual connection using a previously recorded tensor.

**Args:**

- `operation` (str): One of `"add"`, `"concat"`, `"multiply"`, `"gated"`, `"highway"`
- `record_name` (str, optional): Which `Record` to use. If `None`, uses most recent.
- `alpha` (float or LearnableAlpha): Scaling factor for residual branch

**Operations:**

| Operation    | Formula                 | Use case                |
| ------------ | ----------------------- | ----------------------- |
| `add`      | `x + α·r`           | Standard ResNet-style   |
| `concat`   | `cat([x, r], dim=-1)` | DenseNet-style          |
| `multiply` | `x·(1 + α·r)`      | Multiplicative skip     |
| `gated`    | `(1-α)·x + α·r`   | Learnable interpolation |
| `highway`  | `T·x + C·r`         | Highway Networks        |

**Example:**

```python
Apply(operation="gated", record_name="input", alpha=0.5)
```

#### `LearnableAlpha(initial_value, min_value=0.0, max_value=1.0, use_log_space=None)`

Learnable scalar parameter constrained to `[min_value, max_value]`.

**Args:**

- `initial_value` (float): Starting value
- `min_value` (float): Lower bound (inclusive)
- `max_value` (float): Upper bound (inclusive)
- `use_log_space` (bool, optional): Force log or linear parameterization. Auto-detected if `None`.

**Example:**

```python
alpha = LearnableAlpha(0.5, min_value=0.0, max_value=1.0)
x = x + alpha() * residual  # alpha() returns constrained value
```

#### `RecurrentWrapper(module, return_hidden=False)`

Wraps LSTM/GRU modules for seamless integration with `ResidualSequential`.

**Args:**

- `module` (nn.Module): The recurrent module (e.g., `nn.LSTM`)
- `return_hidden` (bool): If `True`, returns `(output, hidden)` tuple

**Example:**

```python
RecurrentWrapper(nn.LSTM(64, 64, batch_first=True), return_hidden=False)
```

---

## Advanced Examples

### Transformer-style block

```python
# Multi-head attention with residual and layer norm
block = ResidualSequential(
    Record(name="input"),
    nn.MultiheadAttention(embed_dim=256, num_heads=8),
    Apply(record_name="input"),
    nn.LayerNorm(256),
  
    Record(name="attn_out"),
    nn.Linear(256, 1024),
    nn.ReLU(),
    nn.Linear(1024, 256),
    Apply(record_name="attn_out"),
    nn.LayerNorm(256),
)
```

### Nested residual blocks

```python
inner_block = ResidualSequential(
    Record(),
    nn.Linear(64, 64),
    nn.ReLU(),
    Apply(),
)

outer_block = ResidualSequential(
    Record(),
    inner_block,
    nn.Linear(64, 64),
    Apply(),
)
```

### Complex encoder block

```python
from collections import OrderedDict

encoder = ResidualSequential(OrderedDict([
    ('record_input', Record(need_projection=True, name="input")),
    ('conv1', nn.Conv1d(64, 128, kernel_size=3, padding=1)),
    ('relu1', nn.ReLU()),
    ('record_mid', Record(name="mid")),
    ('conv2', nn.Conv1d(128, 128, kernel_size=3, padding=1)),
    ('relu2', nn.ReLU()),
    ('apply_long', Apply(record_name="input")),
    ('norm', nn.BatchNorm1d(128)),
    ('conv3', nn.Conv1d(128, 64, kernel_size=1)),
    ('apply_short', Apply(record_name="mid", operation="concat")),
]))
```

---

## Compatibility

### Supported Environments

| Environment                 | Status | Notes                                   |
| --------------------------- | ------ | --------------------------------------- |
| Single GPU training         | ✅     | Full support                            |
| CPU training                | ✅     | Full support                            |
| `nn.DataParallel`         | ✅     | Thread-safe via `threading.local()`   |
| `DistributedDataParallel` | ✅     | Process-safe, recommended for multi-GPU |
| Multi-threaded inference    | ✅     | Safe for Flask/FastAPI servers          |
| Jupyter notebooks           | ✅     | Full support                            |
| `torch.jit.script`        | ❌     | Planned for v1.1                        |
| ONNX export                 | ❌     | Planned for v1.1                        |

### Thread Safety

`torchresidual` uses `threading.local()` for context management, making it safe for:

- `nn.DataParallel` (multiple GPU threads)
- Multi-threaded inference servers
- Concurrent requests in production

See [docs/DESIGN.md](docs/DESIGN.md) for implementation details.

---

## Design Philosophy

### Why thread-local storage?

Traditional approaches store a parent reference in `Apply`, creating circular references:

```
ResidualSequential → Apply → ResidualSequential  # Breaks pickle/deepcopy
```

`torchresidual` uses `threading.local()` to avoid this:

- ✅ No circular references
- ✅ Works with `pickle`, `torch.save`, `deepcopy`
- ✅ Thread-safe for `nn.DataParallel`
- ✅ Clean module hierarchy

### Why tanh parameterization?

`LearnableAlpha` uses `tanh` (not sigmoid) for bounded parameters:

- Better gradient flow near boundaries
- Symmetric around midpoint
- Stable training dynamics

### Why auto-detect log space?

For ranges spanning orders of magnitude (e.g., `1e-4` to `1e-1`), linear space
poorly explores the lower end. Log space provides uniform coverage:

```python
alpha = LearnableAlpha(0.01, min_value=1e-4, max_value=1.0)
# Automatically uses log space (ratio > 100)
```

---

## Examples

See [`examples/`](examples/) directory:

- [`basic_usage.py`](examples/basic_usage.py) - Core concepts
- [`advanced_usage.py`](examples/advanced_usage.py) - Advanced concepts
- [`lstm_residual.py`](examples/lstm_residual.py) - Recurrent networks

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure `pytest` and `mypy` pass
5. Submit a pull request

**Development setup:**

```bash
git clone https://github.com/v-garzon/torchresidual.git
cd torchresidual
pip install -e ".[dev]"
pytest tests/
mypy torchresidual/
```

---

## Citation

If you use `torchresidual` in your research, please cite:

```bibtex
@software{torchresidual2026,
  author = {Garzón, Víctor},
  title = {torchresidual: Flexible residual connections for PyTorch},
  year = {2026},
  url = {https://github.com/v-garzon/torchresidual}
}
```

---

## License

MIT License - see [LICENSE](https://github.com/v-garzon/torchresidual/blob/main/LICENSE) for details.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.
