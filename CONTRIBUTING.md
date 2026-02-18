# Contributing to torchresidual

Thank you for considering contributing to `torchresidual`! This document provides
guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/torchresidual.git
   cd torchresidual
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Verify installation:**
   ```bash
   pytest tests/ -v
   mypy torchresidual/
   ```

## Making Changes

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation only
- `refactor/description` - Code refactoring

### Code Style

- Follow PEP 8
- Use type hints for all public APIs
- Maximum line length: 100 characters
- Use descriptive variable names

### Type Checking

All code must pass mypy:
```bash
mypy torchresidual/
```

Use `# type: ignore[error-code]` sparingly and only when necessary (e.g., PyTorch buffer limitations).

### Testing

1. **Write tests for new features:**
   ```python
   # tests/test_feature.py
   import pytest
   from torchresidual import ResidualSequential
   
   def test_new_feature():
       block = ResidualSequential(...)
       assert block(...).shape == expected_shape
   ```

2. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

3. **Check coverage:**
   ```bash
   pytest tests/ --cov=torchresidual --cov-report=html
   ```

### Documentation

- Add docstrings to all public classes and functions
- Use Google-style docstrings:
  ```python
  def function(arg1: int, arg2: str) -> bool:
      """
      Short description.
      
      Longer description if needed.
      
      Args:
          arg1: Description of arg1.
          arg2: Description of arg2.
          
      Returns:
          Description of return value.
          
      Example::
      
          result = function(42, "hello")
      """
  ```

- Update README.md if adding user-facing features
- Add examples to `examples/` directory for complex features

## Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes and commit:**
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

3. **Push to your fork:**
   ```bash
   git push origin feature/my-new-feature
   ```

4. **Open a pull request:**
   - Provide a clear title and description
   - Reference any related issues
   - Ensure all CI checks pass

### PR Checklist

- [ ] Tests added for new functionality
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Type checking passes (`mypy torchresidual/`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if user-facing change)
- [ ] Code follows project style

## Reporting Bugs

When reporting bugs, please include:

1. **Environment:**
   - Python version
   - PyTorch version
   - torchresidual version
   - OS

2. **Minimal reproducible example:**
   ```python
   import torch
   from torchresidual import ResidualSequential, Record, Apply
   
   # Code that reproduces the bug
   ```

3. **Expected vs actual behavior**

4. **Error message and stack trace**

## Feature Requests

For feature requests, please:

1. Check if the feature already exists or is planned
2. Describe the use case
3. Provide example API (how you'd like to use it)
4. Explain why this feature would be useful

## Code Review Process

- All PRs require at least one review from a maintainer
- Address review comments by pushing additional commits
- Once approved, a maintainer will merge your PR

## Questions?

- Open an issue for questions
- Use the "question" label

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
