#!/bin/bash
# Publish to PyPI (PRODUCTION)

set -e

echo "=========================================="
echo "WARNING: Publishing to PRODUCTION PyPI"
echo "=========================================="
echo
read -p "Are you sure? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

# Check if dist/ exists
if [ ! -d "dist/" ]; then
    echo "Error: dist/ directory not found"
    echo "Run ./scripts/build.sh first"
    exit 1
fi

# Publish
echo "Uploading to pypi.org..."
python -m twine upload dist/*

echo
echo "=========================================="
echo "Published to PyPI!"
echo "=========================================="
echo
echo "Installation:"
echo "  pip install torchresidual"
