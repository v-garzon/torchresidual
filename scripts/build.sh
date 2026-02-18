#!/bin/bash
# Build script for torchresidual

set -e

echo "=========================================="
echo "Building torchresidual v0.1.0"
echo "=========================================="
echo

# Clean previous builds
echo "1. Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info torchresidual.egg-info
echo "   ✓ Cleaned"
echo

# Run tests
echo "2. Running tests..."
python -m pytest tests/ -v --tb=short
echo "   ✓ Tests passed"
echo

# Type check
echo "3. Type checking..."
python -m mypy torchresidual/
echo "   ✓ Type check passed"
echo

# Build distributions
echo "4. Building distributions..."
python -m build
echo "   ✓ Built wheel and sdist"
echo

# Check distributions
echo "5. Checking distributions..."
python -m twine check dist/*
echo "   ✓ Distributions valid"
echo

echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo
echo "Files created:"
ls -lh dist/
echo
echo "Next steps:"
echo "  - Test PyPI: ./scripts/publish-test.sh"
echo "  - Real PyPI: ./scripts/publish.sh"
