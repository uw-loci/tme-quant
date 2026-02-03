#!/usr/bin/env bash

# Minimal Curvelops environment setup for local development.
# Usage: source bin/setup_curvelops_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
UTILS_DIR="${PROJECT_ROOT}/../utils"

FFTW_VERSION="${FFTW_VERSION:-2.1.5}"
CURVELAB_VERSION="${CURVELAB_VERSION:-2.1.2}"

export FFTW="${FFTW:-${UTILS_DIR}/fftw-${FFTW_VERSION}}"
export FDCT="${FDCT:-${UTILS_DIR}/CurveLab-${CURVELAB_VERSION}}"
export CPPFLAGS="-I${FFTW}/include"
export LDFLAGS="-L${FFTW}/lib"

echo "Curvelops environment:"
echo "  FFTW=${FFTW}"
echo "  FDCT=${FDCT}"
echo "  CPPFLAGS=${CPPFLAGS}"
echo "  LDFLAGS=${LDFLAGS}"

if [ ! -d "${FFTW}" ]; then
  echo "WARNING: FFTW directory not found at ${FFTW}"
fi

if [ ! -d "${FDCT}" ]; then
  echo "WARNING: CurveLab directory not found at ${FDCT}"
fi
