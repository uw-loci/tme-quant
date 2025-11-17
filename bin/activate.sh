#!/bin/bash
# Quick activation script for tme-quant-napari-curvealign
# Usage: source bin/activate.sh
#
# This script activates the virtual environment and sets up Curvelops
# environment variables for FFTW and CurveLab.

BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$BIN_DIR")"

# Check if virtual environment exists
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run: bash bin/install.sh"
    return 1 2>/dev/null || exit 1
fi

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Set up Curvelops environment
if [ -f "$BIN_DIR/setup_curvelops_env.sh" ]; then
    source "$BIN_DIR/setup_curvelops_env.sh"
else
    echo "⚠️  Warning: setup_curvelops_env.sh not found"
fi

echo "✅ Environment activated!"
echo "   Project: $PROJECT_ROOT"
echo "   Python: $(python --version 2>/dev/null || echo 'unknown')"
if [ -n "$FFTW" ] && [ -n "$FDCT" ]; then
    echo "   FFTW: $FFTW"
    echo "   CurveLab: $FDCT"
fi
