#!/bin/bash
# Publish to Test PyPI

set -e

echo "=========================================="
echo "Publishing to Test PyPI"
echo "=========================================="
echo

# Check if dist/ exists
if [ ! -d "dist/" ]; then
    echo "Error: dist/ directory not found"
    echo "Run ./scripts/build.sh first"
    exit 1
fi

# Publish
echo "Uploading to test.pypi.org..."
python -m twine upload --repository testpypi dist/*

echo
echo "=========================================="
echo "Published to Test PyPI!"
echo "=========================================="
echo
echo "Test installation:"
echo "  pip install --index-url https://test.pypi.org/simple/ torchresidual"
