#!/bin/bash
# Activate tme-quant environment and set up paths
# Usage: source bin/activate_env.sh

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Initialize conda
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

# Activate environment
conda activate tme-quant

# Set environment variables if needed
if [ -n "$FDCT" ] && [ -d "$FDCT" ]; then
    export FDCT
fi
if [ -n "$FFTW" ] && [ -d "$FFTW" ]; then
    export FFTW
    export CPPFLAGS="-I${FFTW}/include"
    export LDFLAGS="-L${FFTW}/lib"
fi

echo "✅ tme-quant environment activated"
echo "   Project root: $PROJECT_ROOT"

