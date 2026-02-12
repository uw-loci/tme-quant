#!/usr/bin/env bash
################################################################################
# tme-quant Installation Script
#
# Automates installation of tme-quant with curvelops (curvelet backend).
# User prerequisites: 1) clone this repo, 2) download CurveLab to ../utils
#
# Usage: bash bin/install.sh
################################################################################

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
UTILS_DIR="$(cd "$PROJECT_ROOT/../utils" && pwd || echo "$PROJECT_ROOT/../utils")"
PYTHON_VERSION="3.11"
FFTW_VERSION="2.1.5"

# Helpers
print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error()   { echo -e "${RED}[✗]${NC} $1"; }
print_header()  { echo -e "${CYAN}$1${NC}"; }

cdv() {
  # Change directory, verbosely.
  cd "$@" &&
  print_success "Changed directory to: $@"
}

nproc_opt() {
  nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4
}

################################################################################
# Step 1: Ensure uv
################################################################################
ensure_uv() {
  print_header "═══════════════════════════════════════════════════════════════"
  print_header "  Checking for uv"
  print_header "═══════════════════════════════════════════════════════════════"
  echo ""

  if command -v uv &>/dev/null; then
    print_success "uv found: $(uv --version | head -1)"
    return 0
  fi

  print_error "uv not found. Please install it (https://docs.astral.sh/uv/) and try again."
  exit 1
}

################################################################################
# Step 2: Download and build FFTW (to ../utils)
################################################################################
setup_fftw() {
  print_header "═══════════════════════════════════════════════════════════════"
  print_header "  FFTW ${FFTW_VERSION}"
  print_header "═══════════════════════════════════════════════════════════════"
  echo ""

  mkdir -p "$UTILS_DIR"
  UTILS_DIR="$(cd "$UTILS_DIR" && pwd)"
  FFTW_PATH="$UTILS_DIR/fftw-${FFTW_VERSION}"

  if [ -d "$FFTW_PATH" -a -f "$FFTW_PATH/include/fftw.h" ]; then
    print_success "FFTW already built at: $FFTW_PATH"
    export FFTW="$FFTW_PATH"
    return 0
  fi

  # Install build tools if needed
  if ! command -v gcc &>/dev/null || ! command -v make &>/dev/null; then
    print_info "Installing build tools..."
    if [[ "$(uname -s)" == "Darwin" ]]; then
      xcode-select -p &>/dev/null || { print_info "Installing Xcode Command Line Tools (may prompt)..."; xcode-select --install || true; }
    elif command -v apt-get &>/dev/null; then
      sudo apt-get update -qq && sudo apt-get install -y build-essential gcc g++ make curl
    fi
  fi

  cdv "$UTILS_DIR"

  if [ ! -f "fftw-${FFTW_VERSION}.tar.gz" ]; then
    print_info "Downloading FFTW ${FFTW_VERSION}..."
    curl -sL -O "http://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz"
  fi

  if [ ! -d "fftw-${FFTW_VERSION}" ]; then
    print_info "Extracting FFTW..."
    tar xzf "fftw-${FFTW_VERSION}.tar.gz"
  fi

  cdv "fftw-${FFTW_VERSION}"

  # Download newest version of config scripts, to avoid configure failure on modern macOS.
  if [ ! -f config-scripts-downloaded ]; then
    print_info "Updating config scripts..."
    curl -fsL -o config.guess 'https://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.guess;hb=HEAD'
    curl -fsL -o config.sub 'https://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.sub;hb=HEAD'
    chmod +x config.guess config.sub
    touch config-scripts-downloaded
  fi

  print_info "Configuring FFTW (with PIC for shared libs)..."
  (set -x; ./configure --prefix="$(pwd)" --disable-fortran CFLAGS="-fPIC")
  print_info "Building FFTW (this may take several minutes)..."
  (set -x; make -j"$(nproc_opt)" && make install)
  export FFTW="$(pwd)"
  print_success "FFTW built at: $FFTW"
  cdv "$PROJECT_ROOT"
  echo ""
}

################################################################################
# Step 3: Build CurveLab (must be present; fail with instructions if not)
################################################################################
setup_curvelab() {
  print_header "═══════════════════════════════════════════════════════════════"
  print_header "  CurveLab"
  print_header "═══════════════════════════════════════════════════════════════"
  echo ""

  UTILS_DIR="$(cd "$UTILS_DIR" && pwd)"
  FDCT=""

  for d in "$UTILS_DIR"/CurveLab-*; do
    if [ -d "$d" ]; then
      FDCT="$d"
      break
    fi
  done

  if [ -z "$FDCT" -o ! -d "$FDCT" ]; then
    print_error "CurveLab not found in $UTILS_DIR"
    echo ""
    echo "  CurveLab cannot be redistributed. You must:"
    echo "  1. Visit https://curvelet.org/download.php"
    echo "  2. Agree to the license and download CurveLab"
    echo "  3. Extract it to: $UTILS_DIR/"
    echo "     (e.g. $UTILS_DIR/CurveLab-2.1.2)"
    echo ""
    exit 1
  fi

  export FDCT
  print_success "CurveLab found at: $FDCT"

  # Build CurveLab components if needed
  if [ -d "$FDCT/fdct_wrapping_cpp/src" ]; then
    print_info "Building CurveLab fdct_wrapping_cpp..."
    (cd "$FDCT/fdct_wrapping_cpp/src" && make FFTW_DIR="${FFTW:-}") || true
  fi
  if [ -d "$FDCT/fdct/src" ]; then
    print_info "Building CurveLab fdct..."
    (cd "$FDCT/fdct/src" && make FFTW_DIR="${FFTW:-}") || true
  fi
  if [ -d "$FDCT/fdct3d/src" ]; then
    print_info "Building CurveLab fdct3d..."
    (cd "$FDCT/fdct3d/src" && make FFTW_DIR="${FFTW:-}") || true
  fi

  cdv "$PROJECT_ROOT"
  echo ""
}

################################################################################
# Step 4: Create uv env and install
################################################################################
create_env_and_install() {
  print_header "═══════════════════════════════════════════════════════════════"
  print_header "  Creating/updating environment and installing"
  print_header "═══════════════════════════════════════════════════════════════"
  echo ""

  export CPPFLAGS="-I${FFTW}/include"
  export LDFLAGS="-L${FFTW}/lib"

  print_info "Syncing uv environment..."
  uv sync --extra curvelops &&
  print_success "Environment configured."

  print_info "Installing tme-quant..."
  uv pip install -e .

  print_success "Installation complete."
  echo ""
}

################################################################################
# Step 5: Verify installation
################################################################################
verify_installation() {
  print_header "═══════════════════════════════════════════════════════════════"
  print_header "  Verification"
  print_header "═══════════════════════════════════════════════════════════════"
  echo ""

  uv run python -c "
import sys
errors = []
try:
    import pycurvelets as pc
    print('  ✓ pycurvelets')
    print(f'    HAS_CURVELETS = {pc.HAS_CURVELETS}')
except ImportError as e:
    errors.append(f'pycurvelets: {e}')
try:
    import curvelops
    print(f'  ✓ curvelops {curvelops.__version__}')
except ImportError as e:
    errors.append(f'curvelops: {e}')
try:
    import napari_curvealign
    print('  ✓ napari_curvealign')
except ImportError as e:
    errors.append(f'napari_curvealign: {e}')
if errors:
    print('\\n✗ Import errors:')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)
print('\\n✅ All imports successful!')
"

  if [ $? -eq 0 ]; then
    print_success "Verification passed."
  else
    print_error "Verification failed."
    exit 1
  fi
  echo ""
}

################################################################################
# Main
################################################################################
main() {
  echo ""
  print_header "═══════════════════════════════════════════════════════════════"
  print_header "  tme-quant Installation (with curvelops)"
  print_header "═══════════════════════════════════════════════════════════════"
  echo ""
  echo "Prerequisites:"
  echo "  1. Clone this repository"
  echo "  2. Install uv (https://docs.astral.sh/uv/)"
  echo "  2. Download CurveLab to ../utils (see https://curvelet.org/download.php)"
  echo ""
  echo "This script will:"
  echo "  - Download and build FFTW in ../utils"
  echo "  - Detect CurveLab in ../utils then build it"
  echo "  - Create and sync uv env with needed packages"
  echo "  - Run verification to ensure all is functional"
  echo ""
  read -p "Press Enter to continue or Ctrl+C to cancel..."
  echo ""

  ensure_uv
  setup_fftw
  setup_curvelab
  create_env_and_install
  verify_installation

  print_header "═══════════════════════════════════════════════════════════════"
  print_success "Installation complete!"
  print_header "═══════════════════════════════════════════════════════════════"
  echo ""
  echo "Now run:"
  echo ""
  echo "  uv run napari"
  echo ""
  echo "Plugins → napari-curvealign → CurveAlign Widget"
  echo ""
}

main "$@"
