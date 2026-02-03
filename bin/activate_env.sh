#!/usr/bin/env bash

# Activate conda environment for base API development.
# Usage: source bin/activate_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-tme-quant}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is required but not found in PATH."
  echo "Install Miniconda/Anaconda and re-run this script."
  return 1 2>/dev/null || exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate "${ENV_NAME}"

if [ -f "${SCRIPT_DIR}/setup_curvelops_env.sh" ]; then
  # shellcheck disable=SC1090
  source "${SCRIPT_DIR}/setup_curvelops_env.sh"
fi

echo "Environment activated."
echo "  Conda env: ${ENV_NAME}"
