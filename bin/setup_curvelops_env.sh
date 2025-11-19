#!/bin/bash
# Setup script for Curvelops environment variables
# Run this before using Curvelops: source setup_curvelops_env.sh
# 
# NOTE: This is a lightweight helper script for manual setup.
# For automated installation, use: bash setup.sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# Get project root (parent of bin/)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set paths to FFTW and CurveLab installations
# utils/ is parallel to tme-quant/, so we go up one level from project root
# Resolve to absolute paths to avoid issues with relative paths
UTILS_DIR="$(cd "$PROJECT_ROOT/../utils" && pwd 2>/dev/null || echo "$PROJECT_ROOT/../utils")"
export FFTW="${UTILS_DIR}/fftw-2.1.5"
export FDCT="${UTILS_DIR}/CurveLab-2.1.2"

# Check if directories exist and provide helpful messages
if [ ! -d "$FFTW" ]; then
    echo "⚠️  Warning: FFTW directory not found at $FFTW"
    echo "   Run 'bash setup.sh' for automated installation or manually set FFTW path."
fi

if [ ! -d "$FDCT" ]; then
    echo "⚠️  Warning: CurveLab directory not found at $FDCT"
    echo "   Download CurveLab from: https://curvelet.org/download.php"
    echo "   Then set FDCT environment variable to the CurveLab directory."
fi

# Set compiler flags for building
export CPPFLAGS="-I${FFTW}/include"
export LDFLAGS="-L${FFTW}/lib"

echo ""
echo "Curvelops environment variables set:"
echo "  FFTW: $FFTW"
echo "  FDCT: $FDCT"
echo "  CPPFLAGS: $CPPFLAGS"
echo "  LDFLAGS: $LDFLAGS"
echo ""

if [ -d "$FFTW" ] && [ -d "$FDCT" ]; then
    echo "✅ Dependencies found! You can now install Curvelops with:"
    echo "   pip install -e \".[curvelops]\""
else
    echo "❌ Please install missing dependencies first."
    echo "   See SETUP_GUIDE.md for detailed instructions."
    echo "   Or run: bash setup.sh"
fi
echo ""
