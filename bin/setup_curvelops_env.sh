#!/bin/bash
# Setup script for Curvelops environment variables
# Run this before using Curvelops: source setup_curvelops_env.sh
# 
# NOTE: This is a lightweight helper script for manual setup.
# For automated installation, use: bash setup.sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# Set paths to FFTW and CurveLab installations (relative to project root)
# Adjust these paths based on your directory structure
export FFTW="${SCRIPT_DIR}/../utils/fftw-2.1.5"
export FDCT="${SCRIPT_DIR}/../utils/CurveLab-2.1.2"

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
