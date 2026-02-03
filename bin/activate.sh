#!/usr/bin/env bash

# Activate local virtual environment and Curvelops paths.
# Usage: source bin/activate.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
VENV_DIR="${PROJECT_ROOT}/.venv"

if [ ! -d "${VENV_DIR}" ]; then
  echo "ERROR: Virtual environment not found at ${VENV_DIR}."
  echo "Run: bash bin/install.sh"
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

if [ -f "${SCRIPT_DIR}/setup_curvelops_env.sh" ]; then
  # shellcheck disable=SC1090
  source "${SCRIPT_DIR}/setup_curvelops_env.sh"
fi

echo "Environment activated."
echo "  Project: ${PROJECT_ROOT}"
echo "  Python: $(python --version 2>/dev/null || echo 'unknown')"
