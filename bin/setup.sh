#!/usr/bin/env bash

# Minimal local setup for base API development (conda-based).
# Usage: bash bin/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

ENV_NAME="${ENV_NAME:-tme-quant}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
INSTALL_DEV="${INSTALL_DEV:-1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is required but not found in PATH."
  echo "Install Miniconda/Anaconda and re-run this script."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "== Using existing conda env: ${ENV_NAME} =="
else
  echo "== Creating conda env: ${ENV_NAME} (Python ${PYTHON_VERSION}) =="
  conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}" pip
fi

conda activate "${ENV_NAME}"

echo "== Upgrading build tools =="
python -m pip install --upgrade pip

if [ -f "${SCRIPT_DIR}/setup_curvelops_env.sh" ]; then
  # shellcheck disable=SC1090
  source "${SCRIPT_DIR}/setup_curvelops_env.sh"
fi

EXTRAS=()
if [ "${INSTALL_DEV}" = "1" ]; then
  EXTRAS+=("dev")
fi
if [ -n "${FFTW:-}" ] && [ -d "${FFTW}" ] && [ -n "${FDCT:-}" ] && [ -d "${FDCT}" ]; then
  EXTRAS+=("curvelab")
else
  echo "NOTE: FFTW/CurveLab not found; skipping curvelab extra."
fi

if [ "${#EXTRAS[@]}" -gt 0 ]; then
  EXTRAS_CSV="$(IFS=,; echo "${EXTRAS[*]}")"
  echo "== Installing tme-quant with extras: ${EXTRAS_CSV} =="
  python -m pip install -e ".[${EXTRAS_CSV}]"
else
  echo "== Installing tme-quant base dependencies =="
  python -m pip install -e .
fi

echo "Done. Activate with: source bin/activate_env.sh"
