#!/usr/bin/env bash

# Minimal local install for napari plugin development.
# Usage: bash bin/install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
VENV_DIR="${PROJECT_ROOT}/.venv"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required but not found in PATH."
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "== Creating virtual environment at ${VENV_DIR} =="
  python3 -m venv "${VENV_DIR}"
else
  echo "== Using existing virtual environment at ${VENV_DIR} =="
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "== Upgrading build tools =="
python -m pip install --upgrade pip setuptools wheel

if [ -f "${SCRIPT_DIR}/setup_curvelops_env.sh" ]; then
  # shellcheck disable=SC1090
  source "${SCRIPT_DIR}/setup_curvelops_env.sh"
fi

EXTRAS=()
if [ -n "${FFTW:-}" ] && [ -d "${FFTW}" ] && [ -n "${FDCT:-}" ] && [ -d "${FDCT}" ]; then
  EXTRAS+=("curvelab")
else
  echo "NOTE: FFTW/CurveLab not found; installing without curvelab extra."
fi

if [ "${#EXTRAS[@]}" -gt 0 ]; then
  EXTRAS_CSV="$(IFS=,; echo "${EXTRAS[*]}")"
  echo "== Installing tme-quant with extras: ${EXTRAS_CSV} =="
  python -m pip install -e ".[${EXTRAS_CSV}]"
else
  echo "== Installing tme-quant base dependencies =="
  python -m pip install -e .
fi

echo "Done. Activate with: source bin/activate.sh"
