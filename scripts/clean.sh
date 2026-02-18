#!/bin/bash
# Clean build artifacts

echo "Cleaning build artifacts..."
rm -rf build/ dist/ *.egg-info torchresidual.egg-info
rm -rf .pytest_cache/ .mypy_cache/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "✓ Cleaned"
