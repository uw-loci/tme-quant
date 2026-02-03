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
if [ -n "${FFTW:-}" ]; then
  if [ ! -d "${FFTW}" ]; then
    echo "::error title=Invalid FFTW path::FFTW is set to '${FFTW}' but the directory does not exist."
    exit 1
  fi
  FFTW_PATH="${FFTW}"
  echo "Using pre-installed FFTW at ${FFTW_PATH}"
else
  if [ ! -d "fftw-${FFTW_VERSION}" ]; then
    if [ ! -f "fftw-${FFTW_VERSION}.tar.gz" ]; then
      curl -fsSL "http://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz" -o "fftw-${FFTW_VERSION}.tar.gz"
    fi
    tar xzf "fftw-${FFTW_VERSION}.tar.gz"
    pushd "fftw-${FFTW_VERSION}" >/dev/null
    CFLAGS="-fPIC" CXXFLAGS="-fPIC" ./configure --prefix="$(pwd)" --disable-fortran --enable-shared --disable-static
    make -j"$(nproc)"
    make install
    popd >/dev/null
    ln -sf "fftw-${FFTW_VERSION}" fftw
  else
    echo "FFTW already present, skipping rebuild."
  fi
  FFTW_PATH="${UTILS_DIR}/fftw-${FFTW_VERSION}"
fi

# Some CurveLab makefiles expect headers/libs under FFTW_DIR/fftw
mkdir -p "${FFTW_PATH}/fftw"
if [ -f "${FFTW_PATH}/include/fftw.h" ]; then
  ln -sf "${FFTW_PATH}/include/fftw.h" "${FFTW_PATH}/fftw/fftw.h"
fi
if [ -d "${FFTW_PATH}/lib" ]; then
  ln -sf "${FFTW_PATH}/lib/"* "${FFTW_PATH}/fftw/" || true
fi

ARCHIVE_NAME="$(basename "${RESTRICTED_URL:-CurveLab-${CURVELAB_VERSION}.tar.gz}")"
DEFAULT_ARCHIVE="CurveLab-${CURVELAB_VERSION}.tar.gz"

if [ -n "${FDCT:-}" ]; then
  if [ ! -d "${FDCT}" ]; then
    echo "::error title=Invalid CurveLab path::FDCT is set to '${FDCT}' but the directory does not exist."
    exit 1
  fi
  FDCT_PATH="${FDCT}"
  echo "Using pre-installed CurveLab at ${FDCT_PATH}"
else
  echo "== Preparing CurveLab ${CURVELAB_VERSION} archive =="
  if [ -f "${DEFAULT_ARCHIVE}" ]; then
    ARCHIVE_NAME="${DEFAULT_ARCHIVE}"
  elif [ ! -f "${ARCHIVE_NAME}" ]; then
    if [ -n "${RESTRICTED_USER:-}" ] && [ -n "${RESTRICTED_PASSWORD:-}" ] && [ -n "${RESTRICTED_URL:-}" ]; then
      curl --fail-with-body -fL --retry 3 --retry-all-errors --connect-timeout 20 --max-time 600 \
        -A "Mozilla/5.0 (GitHub Actions)" \
        -u "${RESTRICTED_USER}:${RESTRICTED_PASSWORD}" \
        -O "${RESTRICTED_URL}"
    else
      echo "::error title=CurveLab archive missing::Provide FETCH_FDCT secret (preferred) or define RESTRICTED_USER/RESTRICTED_PASSWORD/RESTRICTED_URL."
      exit 1
    fi
  fi

  if [ ! -d "CurveLab-${CURVELAB_VERSION}" ]; then
    echo "== Extracting ${ARCHIVE_NAME} =="
    if command -v file >/dev/null 2>&1; then
      MIME_TYPE=$(file -b --mime-type "${ARCHIVE_NAME}")
    else
      MIME_TYPE=""
    fi
    if [[ "${MIME_TYPE}" == text/html* ]]; then
      echo "::error title=CurveLab download failed::${ARCHIVE_NAME} is HTML (likely an auth error or redirect)."
      if command -v head >/dev/null 2>&1; then
        echo "First 40 lines of response:"
        head -n 40 "${ARCHIVE_NAME}"
      fi
      echo "Ensure FETCH_FDCT runs a curl command that downloads the actual CurveLab-${CURVELAB_VERSION} archive."
      exit 1
    fi
    case "${MIME_TYPE}" in
      application/gzip|application/x-gzip)
        tar xzf "${ARCHIVE_NAME}"
        ;;
      application/x-tar|"")
        # Fallback to plain tar if mime type unknown
        tar xf "${ARCHIVE_NAME}"
        ;;
      application/zip)
        unzip -q "${ARCHIVE_NAME}"
        ;;
      *)
        echo "::error title=Unknown archive format::Don't know how to extract ${ARCHIVE_NAME} (${MIME_TYPE})."
        exit 1
        ;;
    esac
  fi

  echo "== Building CurveLab binaries =="
  export FFTW_DIR="${FFTW_PATH}"
  export FFTW="${FFTW_PATH}"
  export CPPFLAGS="-I${FFTW_PATH}/include"
  export LDFLAGS="-L${FFTW_PATH}/lib"
  pushd "CurveLab-${CURVELAB_VERSION}" >/dev/null
  pushd fdct_wrapping_cpp/src >/dev/null
  make FFTW_DIR="${FFTW_PATH}" FFTW="${FFTW_PATH}" FFTW_INC="${FFTW_PATH}/include" FFTW_INCLUDE="${FFTW_PATH}/include" FFTW_LIB="${FFTW_PATH}/lib" FFTW_LIBDIR="${FFTW_PATH}/lib"
  popd >/dev/null
  pushd fdct3d/src >/dev/null
  make FFTW_DIR="${FFTW_PATH}" FFTW="${FFTW_PATH}" FFTW_INC="${FFTW_PATH}/include" FFTW_INCLUDE="${FFTW_PATH}/include" FFTW_LIB="${FFTW_PATH}/lib" FFTW_LIBDIR="${FFTW_PATH}/lib"
  popd >/dev/null
  popd >/dev/null

  FDCT_WRAPPING_LIB="CurveLab-${CURVELAB_VERSION}/fdct_wrapping_cpp/src/libfdct_wrapping.a"
  FDCT3D_LIB="CurveLab-${CURVELAB_VERSION}/fdct3d/src/libfdct3d.a"
  if [ ! -f "${FDCT_WRAPPING_LIB}" ]; then
    echo "::error title=CurveLab build missing::libfdct_wrapping.a not found after build."
    echo "Contents of fdct_wrapping_cpp/src:"
    ls -la "CurveLab-${CURVELAB_VERSION}/fdct_wrapping_cpp/src" || true
    exit 1
  fi
  if [ ! -f "${FDCT3D_LIB}" ]; then
    echo "::error title=CurveLab build missing::libfdct3d.a not found after build."
    echo "Contents of fdct3d/src:"
    ls -la "CurveLab-${CURVELAB_VERSION}/fdct3d/src" || true
    exit 1
  fi

  FDCT_PATH="${UTILS_DIR}/CurveLab-${CURVELAB_VERSION}"
fi

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

