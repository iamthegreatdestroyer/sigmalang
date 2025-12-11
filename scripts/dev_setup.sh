#!/bin/bash
# ΣLANG Development Environment Setup
# Run this to set up your local development environment

set -e

echo "=========================================="
echo "ΣLANG Development Setup"
echo "=========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python 3.9 or higher is required (found $PYTHON_VERSION)"
    exit 1
fi
echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
echo "✓ Virtual environment ready"

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "Installing dependencies..."
pip install numpy --quiet
pip install pytest pytest-cov --quiet

# Install package in development mode
echo "Installing ΣLANG in development mode..."
pip install -e . --quiet

echo "✓ Dependencies installed"

# Run tests
echo ""
echo "Running tests..."
python -m pytest tests/ -v --tb=short

# Run demo
echo ""
echo "Running compression demo..."
python tests/test_sigmalang.py --demo

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  python -m pytest tests/ -v"
echo ""
echo "To run the demo:"
echo "  python tests/test_sigmalang.py --demo"
echo ""
echo "To train the codebook:"
echo "  python training/train.py --mode bootstrap"
echo ""
