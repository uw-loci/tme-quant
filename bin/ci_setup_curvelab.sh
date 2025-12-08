#!/usr/bin/env bash

# CI helper: build FFTW 2.1.5 and CurveLab 2.1.3 from source and export env vars.
# Requires three secrets: RESTRICTED_USER, RESTRICTED_PASSWORD, RESTRICTED_URL.

set -euo pipefail

FFTW_VERSION="${FFTW_VERSION:-2.1.5}"
CURVELAB_VERSION="${CURVELAB_VERSION:-2.1.3}"
CURVELAB_ARCHIVE="${CURVELAB_ARCHIVE:-CurveLab-${CURVELAB_VERSION}.tar.gz}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UTILS_DIR="${ROOT_DIR}/utils"

mkdir -p "${UTILS_DIR}"
cd "${UTILS_DIR}"

echo "== Preparing FFTW ${FFTW_VERSION} =="
if [ ! -d "fftw-${FFTW_VERSION}" ]; then
  if [ ! -f "fftw-${FFTW_VERSION}.tar.gz" ]; then
    curl -fsSL "http://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz" -o "fftw-${FFTW_VERSION}.tar.gz"
  fi
  tar xzf "fftw-${FFTW_VERSION}.tar.gz"
  pushd "fftw-${FFTW_VERSION}" >/dev/null
  ./configure --prefix="$(pwd)" --disable-fortran
  make -j"$(nproc)"
  make install
  popd >/dev/null
  ln -sf "fftw-${FFTW_VERSION}" fftw
else
  echo "FFTW already present, skipping rebuild."
fi

ARCHIVE_NAME="$(basename "${RESTRICTED_URL:-CurveLab-${CURVELAB_VERSION}.tar.gz}")"
DEFAULT_ARCHIVE="CurveLab-${CURVELAB_VERSION}.tar.gz"

echo "== Preparing CurveLab ${CURVELAB_VERSION} archive =="
if [ -f "${DEFAULT_ARCHIVE}" ]; then
  ARCHIVE_NAME="${DEFAULT_ARCHIVE}"
elif [ ! -f "${ARCHIVE_NAME}" ]; then
  if [ -n "${RESTRICTED_USER:-}" ] && [ -n "${RESTRICTED_PASSWORD:-}" ] && [ -n "${RESTRICTED_URL:-}" ]; then
    curl -fL -u "${RESTRICTED_USER}:${RESTRICTED_PASSWORD}" -O "${RESTRICTED_URL}"
  else
    echo "::error title=CurveLab archive missing::Provide FETCH_FDCT secret (preferred) or define RESTRICTED_USER/RESTRICTED_PASSWORD/RESTRICTED_URL."
    exit 1
  fi
fi

if [ ! -d "CurveLab-${CURVELAB_VERSION}" ]; then
  tar xzf "${ARCHIVE_NAME}"
fi

echo "== Building CurveLab binaries =="
pushd "CurveLab-${CURVELAB_VERSION}" >/dev/null
pushd fdct_wrapping_cpp/src >/dev/null
make
popd >/dev/null
pushd fdct3d/src >/dev/null
make
popd >/dev/null
popd >/dev/null

FFTW_PATH="${UTILS_DIR}/fftw-${FFTW_VERSION}"
FDCT_PATH="${UTILS_DIR}/CurveLab-${CURVELAB_VERSION}"

echo "FFTW path: ${FFTW_PATH}"
echo "CurveLab path: ${FDCT_PATH}"

if [ -n "${GITHUB_ENV:-}" ]; then
  {
    echo "FFTW=${FFTW_PATH}"
    echo "FDCT=${FDCT_PATH}"
    echo "CPPFLAGS=-I${FFTW_PATH}/include"
    echo "LDFLAGS=-L${FFTW_PATH}/lib"
  } >> "${GITHUB_ENV}"
else
  export FFTW="${FFTW_PATH}"
  export FDCT="${FDCT_PATH}"
  export CPPFLAGS="-I${FFTW_PATH}/include"
  export LDFLAGS="-L${FFTW_PATH}/lib"
fi

echo "CurveLab setup complete."

