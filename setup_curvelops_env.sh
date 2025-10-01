#!/bin/bash
# Setup script for Curvelops environment variables
# Run this before using Curvelops: source setup_curvelops_env.sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# Set paths to FFTW and CurveLab installations (relative to project root)
# Adjust these paths based on your directory structure
export FFTW="${SCRIPT_DIR}/../utils/fftw-2.1.5"
export FDCT="${SCRIPT_DIR}/../utils/CurveLab-2.1.2"

# Check if directories exist and provide helpful messages
if [ ! -d "$FFTW" ]; then
    echo "Warning: FFTW directory not found at $FFTW"
    echo "Please adjust the FFTW path in this script or install FFTW to the expected location."
fi

if [ ! -d "$FDCT" ]; then
    echo "Warning: CurveLab directory not found at $FDCT"
    echo "Please adjust the FDCT path in this script or install CurveLab to the expected location."
fi

# Set compiler flags for building
export CPPFLAGS="-I${FFTW}/include"
export LDFLAGS="-L${FFTW}/lib"

echo "Curvelops environment variables set:"
echo "  FFTW: $FFTW"
echo "  FDCT: $FDCT"
echo "  CPPFLAGS: $CPPFLAGS"
echo "  LDFLAGS: $LDFLAGS"
echo ""

if [ -d "$FFTW" ] && [ -d "$FDCT" ]; then
    echo "✅ Dependencies found! You can now install Curvelops with:"
    echo "  pip install -e \".[curvelops]\""
else
    echo "❌ Please install missing dependencies first."
    echo "See CURVELOPS_INTEGRATION.md for detailed instructions."
fi
