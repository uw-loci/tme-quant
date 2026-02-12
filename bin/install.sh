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
UTILS_DIR="$(cd "$PROJECT_ROOT/../utils" 2>/dev/null && pwd || echo "$PROJECT_ROOT/../utils")"
ENV_NAME="tme-quant"
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
# Step 1: Ensure conda or mamba (install Miniforge if needed)
################################################################################
ensure_conda() {
  print_header "═══════════════════════════════════════════════════════════════"
  print_header "  Checking for conda/mamba"
  print_header "═══════════════════════════════════════════════════════════════"
  echo ""

  if command -v mamba &>/dev/null; then
    print_success "mamba found: $(mamba --version 2>/dev/null | head -1 || true)"
    CONDA_CMD="mamba"
    return 0
  fi
  if command -v conda &>/dev/null; then
    print_success "conda found: $(conda --version 2>/dev/null || true)"
    CONDA_CMD="conda"
    return 0
  fi

  print_warning "Neither conda nor mamba found. Installing Miniforge (includes mamba)..."
  install_miniforge
  CONDA_CMD="mamba"
}

install_miniforge() {
  local os arch installer_url install_dir="$HOME/miniforge3"
  case "$(uname -s)" in
    Darwin) os="MacOSX" ;;
    Linux)  os="Linux" ;;
    *) print_error "Unsupported OS: $(uname -s)"; exit 1 ;;
  esac

  case "$(uname -m)" in
    x86_64)  arch="x86_64" ;;
    arm64)   arch="arm64" ;;
    aarch64) arch="aarch64" ;;
    *) print_error "Unsupported arch: $(uname -m)"; exit 1 ;;
  esac

  # Map Darwin arm64 -> MacOSX arm64; Linux aarch64 ok
  [ "$os" = "MacOSX" ] && [ "$arch" = "aarch64" ] && arch="arm64"

  # Use a known stable version (avoids API parsing / grep -oP portability)
  local version="26.1.0-0"
  installer_url="https://github.com/conda-forge/miniforge/releases/download/${version}/Miniforge3-${version}-${os}-${arch}.sh"

  print_info "Downloading Miniforge..."
  local tmp_installer
  tmp_installer=$(mktemp)
  curl -sL -o "$tmp_installer" "$installer_url" || {
    print_error "Failed to download Miniforge from $installer_url"
    exit 1
  }

  print_info "Installing Miniforge to $install_dir..."
  bash "$tmp_installer" -b -p "$install_dir"
  rm -f "$tmp_installer"

  # Initialize for current shell
  set +u
  # shellcheck source=/dev/null
  source "$install_dir/etc/profile.d/conda.sh" 2>/dev/null || true
  set -u
  export PATH="$install_dir/bin:$PATH"

  print_success "Miniforge installed. Run 'conda init bash' (or zsh) and open a new shell for persistence."
  echo ""
}

################################################################################
# Step 2: Fail fast if environment already exists
################################################################################
check_env_not_exists() {
  print_header "═══════════════════════════════════════════════════════════════"
  print_header "  Checking environment"
  print_header "═══════════════════════════════════════════════════════════════"
  echo ""

  # Ensure conda is in PATH for this check
  if [ -z "${CONDA_CMD:-}" ]; then
    ensure_conda
  fi

  for init in "$HOME/miniforge3/etc/profile.d/conda.sh" \
              "$HOME/miniconda3/etc/profile.d/conda.sh" \
              "$HOME/anaconda3/etc/profile.d/conda.sh" \
              "/opt/conda/etc/profile.d/conda.sh"; do
    if [ -f "$init" ]; then
      set +u
      # shellcheck source=/dev/null
      source "$init" 2>/dev/null || true
      set -u
      break
    fi
  done

  if conda env list 2>/dev/null | grep -qE "^\s*${ENV_NAME}\s"; then
    print_error "Conda environment '${ENV_NAME}' already exists."
    echo ""
    echo "  To avoid conflicts, remove it first:"
    echo "    conda env remove -n ${ENV_NAME}"
    echo ""
    echo "  Or use a different environment by editing ENV_NAME in bin/install.sh"
    exit 1
  fi
  print_success "Environment '${ENV_NAME}' does not exist; proceeding."
  echo ""
}

################################################################################
# Step 3: Download and build FFTW (to ../utils)
################################################################################
setup_fftw() {
  print_header "═══════════════════════════════════════════════════════════════"
  print_header "  FFTW ${FFTW_VERSION}"
  print_header "═══════════════════════════════════════════════════════════════"
  echo ""

  mkdir -p "$UTILS_DIR"
  UTILS_DIR="$(cd "$UTILS_DIR" && pwd)"
  FFTW_PATH="$UTILS_DIR/fftw-${FFTW_VERSION}"

  if [ -d "$FFTW_PATH" ] && [ -f "$FFTW_PATH/include/fftw.h" ]; then
    print_success "FFTW already built at: $FFTW_PATH"
    export FFTW="$FFTW_PATH"
    return 0
  fi

  # Install build tools if needed
  if ! command -v gcc &>/dev/null || ! command -v make &>/dev/null; then
    print_info "Installing build tools..."
    if [[ "$(uname -s)" == "Darwin" ]]; then
      xcode-select -p &>/dev/null || { print_info "Installing Xcode Command Line Tools (may prompt)..."; xcode-select --install 2>/dev/null || true; }
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
# Step 4: Detect CurveLab (must be present; fail with instructions if not)
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

  if [ -z "$FDCT" ] || [ ! -d "$FDCT" ]; then
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
  if [ -d "$FDCT/fdct_wrapping_cpp/src" ] && [ ! -f "$FDCT/fdct_wrapping_cpp/src/fdct_wrapping"* ]; then
    print_info "Building CurveLab fdct_wrapping_cpp..."
    (cd "$FDCT/fdct_wrapping_cpp/src" && make FFTW_DIR="${FFTW:-}" 2>/dev/null) || true
  fi
  if [ -d "$FDCT/fdct/src" ]; then
    print_info "Building CurveLab fdct..."
    (cd "$FDCT/fdct/src" && make FFTW_DIR="${FFTW:-}" 2>/dev/null) || true
  fi
  if [ -d "$FDCT/fdct3d/src" ]; then
    print_info "Building CurveLab fdct3d..."
    (cd "$FDCT/fdct3d/src" && make FFTW_DIR="${FFTW:-}" 2>/dev/null) || true
  fi

  cdv "$PROJECT_ROOT"
  echo ""
}

################################################################################
# Step 5: Create conda env and install
################################################################################
create_env_and_install() {
  print_header "═══════════════════════════════════════════════════════════════"
  print_header "  Creating environment and installing"
  print_header "═══════════════════════════════════════════════════════════════"
  echo ""

  export CPPFLAGS="-I${FFTW}/include"
  export LDFLAGS="-L${FFTW}/lib"

  print_info "Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
  $CONDA_CMD create -y -n "$ENV_NAME" python="$PYTHON_VERSION" pip -c conda-forge

  set +u
  for init in "$HOME/miniforge3/etc/profile.d/conda.sh" \
              "$HOME/miniconda3/etc/profile.d/conda.sh" \
              "$HOME/anaconda3/etc/profile.d/conda.sh" \
              "/opt/conda/etc/profile.d/conda.sh"; do
    if [ -f "$init" ]; then
      source "$init"
      break
    fi
  done
  conda activate "$ENV_NAME"
  set -u

  print_info "Installing build tools for curvelops..."
  pip install -q pybind11 scikit-build-core cmake ninja 2>/dev/null || true
  $CONDA_CMD install -y -c conda-forge pybind11 cmake ninja 2>/dev/null || true

  print_info "Installing curvelops (this may take several minutes)..."
  pip install -v "curvelops @ git+https://github.com/PyLops/curvelops@0.23" || {
    print_error "curvelops installation failed. Check FFTW and CurveLab paths."
    exit 1
  }

  print_info "Installing napari and Qt..."
  $CONDA_CMD install -y -c conda-forge napari pyqt qtpy

  print_info "Installing tme-quant..."
  pip install -e .

  print_success "Installation complete."
  echo ""
}

################################################################################
# Step 6: Verify installation
################################################################################
verify_installation() {
  print_header "═══════════════════════════════════════════════════════════════"
  print_header "  Verification"
  print_header "═══════════════════════════════════════════════════════════════"
  echo ""

  python -c "
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
  echo "  2. Download CurveLab to ../utils (see https://curvelet.org/download.php)"
  echo ""
  echo "This script will:"
  echo "  - Install Miniforge (if conda/mamba not found)"
  echo "  - Download and build FFTW in ../utils"
  echo "  - Detect CurveLab in ../utils"
  echo "  - Create conda env '${ENV_NAME}' and install with curvelops"
  echo "  - Run verification"
  echo ""
  read -p "Press Enter to continue or Ctrl+C to cancel..."
  echo ""

  ensure_conda
  check_env_not_exists
  setup_fftw
  setup_curvelab
  create_env_and_install
  verify_installation

  print_header "═══════════════════════════════════════════════════════════════"
  print_success "Installation complete!"
  print_header "═══════════════════════════════════════════════════════════════"
  echo ""
  echo "Activate the environment with:"
  echo ""
  echo "  conda activate ${ENV_NAME}"
  echo ""
  echo "Then launch Napari:"
  echo ""
  echo "  napari"
  echo ""
  echo "Plugins → napari-curvealign → CurveAlign Widget"
  echo ""
}

main "$@"
