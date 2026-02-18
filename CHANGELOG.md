# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-18

Initial release.

### Added

#### Core Components
- `ResidualSequential`: Drop-in replacement for `nn.Sequential` with residual connection support
- `Record`: Module to save tensor state at specific points in the forward pass
- `Apply`: Module to apply residual connections with five operation types:
  - `add`: Standard ResNet-style addition (default)
  - `concat`: DenseNet-style concatenation
  - `multiply`: Multiplicative residual
  - `gated`: Learnable interpolation between paths
  - `highway`: Highway Networks with dual learnable gates

#### Advanced Features
- `LearnableAlpha`: Constrained learnable scalar with:
  - Tanh-based reparameterization for smooth gradients
  - Automatic log/linear space detection (>100× range → log space)
  - Bounds enforcement via transformation, not clamping
- `RecurrentWrapper`: Integration wrapper for LSTM/GRU modules
- Automatic shape projection when `need_projection=True` in `Record`
- Named record points for complex multi-skip architectures
- Thread-safe execution via `threading.local()` (compatible with `nn.DataParallel`)
- Enhanced `__repr__` with proper indentation for nested blocks

#### Developer Experience
- Full type hints with mypy validation
- Comprehensive test suite (>40 tests, 100% pass rate)
- Detailed documentation and examples
- CI/CD pipeline with Python 3.8-3.11 support
- PEP 561 type marker (`py.typed`) for IDE support

### Design Decisions
- Thread-local storage to avoid circular references
- Tanh parameterization for bounded parameters
- Log-space auto-detection for wide-range parameters
- Default operation `"add"` for intuitive ResNet-style usage

### Known Limitations
- TorchScript/ONNX export not supported (planned for v1.1)
- Requires PyTorch ≥1.9

---

## [Unreleased]

### Planned for v1.1
- TorchScript support via `ResidualSequentialScript`
- ONNX export compatibility
- Additional utility functions for common residual patterns
- Performance benchmarks
- Extended documentation with more use cases
