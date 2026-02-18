"""Setup script for torchresidual."""
from setuptools import setup, find_packages

setup(
    name="torchresidual",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "mypy",
            "numpy",
        ],
    },
)
