#!/usr/bin/env bash
################################################################################
# tme-quant Installation Script
#
# Automates installation of tme-quant with curvelops (curvelet backend).
# User prerequisite: download CurveLab into this repo's utils/ directory.
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
UTILS_DIR="$PROJECT_ROOT/utils"
PYTHON_VERSION="3.11"
FFTW_VERSION="2.1.5"

# Helpers
print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error()   { echo -e "${RED}[✗]${NC} $1"; }
print_header()  { echo -e "${CYAN}$1${NC}"; }

report_failure() {
  local status=$?
  print_error "Installation stopped (exit code $status). Review the error above."
  exit "$status"
}
trap report_failure ERR

# Fancy way to calculate cores for multithread build
nproc_opt() {
  nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4
}

# MinGW make can expand Git Bash's /bin/sh to an unquoted Windows path with spaces.
make_with_shell() {
  case "$(uname -s)" in
    MINGW*|MSYS*)
      if ! command -v cygpath >/dev/null 2>&1; then
        print_error "cygpath is required to select a safe Git Bash shell path."
        return 1
      fi
      local shell_path
      if ! shell_path="$(cygpath -s -m /usr/bin/sh)" || [ -z "$shell_path" ] || [[ "$shell_path" == *" "* ]]; then
        print_error "Could not get a space-free path to Git Bash sh.exe; use WSL2 or a toolchain with a space-free shell path."
        return 1
      fi
      print_info "Using space-free make shell: $shell_path"
      make "SHELL=$shell_path" "$@"
      ;;
    *)
      make "$@"
      ;;
  esac
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

ensure_build_tools() {
  local missing=0
  for tool in curl tar make gcc; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      print_error "Required build tool not found: $tool"
      missing=1
    fi
  done
  if [ "$missing" -ne 0 ]; then
    print_info "Install the GNU build tools for your platform, or use WSL2 on Windows."
    return 1
  fi
}

ensure_native_python_compiler() {
  case "$(uname -s)" in
    MINGW*|MSYS*)
      if ! command -v cl >/dev/null 2>&1; then
        print_error "Native Windows curvelops builds require MSVC (cl.exe); MinGW gcc is not a substitute."
        print_info "Install Visual Studio C++ Build Tools and launch Git Bash from a developer environment, or use WSL2."
        print_warning "The GNU-built FFTW and CurveLab libraries may also need MSVC-compatible builds."
        return 1
      fi
      ;;
  esac
}

has_curvelab_sources() {
  [ -f "$1/fdct_wrapping_cpp/src/makefile" ] && [ -f "$1/fdct3d/src/makefile" ]
}

check_curvelab() {
  local -a candidates=()
  local candidate nested
  for candidate in "$UTILS_DIR"/CurveLab-*; do
    [ -d "$candidate" ] || continue
    if has_curvelab_sources "$candidate"; then
      candidates+=("$candidate")
      continue
    fi
    for nested in "$candidate"/*; do
      [ -d "$nested" ] || continue
      if has_curvelab_sources "$nested"; then
        candidates+=("$nested")
      fi
    done
  done

  if [ "${#candidates[@]}" -gt 1 ]; then
    print_error "More than one CurveLab source tree found in $UTILS_DIR; keep only the version you want to build."
    return 1
  fi
  if [ "${#candidates[@]}" -eq 0 ]; then
    print_error "CurveLab source not found in $UTILS_DIR"
    echo "  Expected fdct_wrapping_cpp/src/makefile and fdct3d/src/makefile under CurveLab-<version>."
    echo "  Download it from https://curvelet.org/download.php and extract it into $UTILS_DIR"
    return 1
  fi

  FDCT="${candidates[0]}"
  export FDCT
  print_success "CurveLab source found at: $FDCT"
}

################################################################################
# Step 2: Download and build FFTW (to this repo's utils/)
################################################################################
setup_fftw() {
  print_header "═══════════════════════════════════════════════════════════════"
  print_header "  FFTW ${FFTW_VERSION}"
  print_header "═══════════════════════════════════════════════════════════════"
  echo ""

  mkdir -p "$UTILS_DIR"
  FFTW_PATH="$UTILS_DIR/fftw-${FFTW_VERSION}"
  FFTW_ARCHIVE="$UTILS_DIR/fftw-${FFTW_VERSION}.tar.gz"

  if [ -d "$FFTW_PATH" ] && [ -f "$FFTW_PATH/include/fftw.h" ] && [ -f "$FFTW_PATH/lib/libfftw.a" ]; then
    # Verify FFTW was built in double precision (not --enable-float).
    # Double-precision FFTW exports 'fftw_create_plan'; float-precision exports 'sfftw_create_plan'.
    if ! command -v nm >/dev/null 2>&1; then
      print_warning "nm not found; reusing existing FFTW without checking its precision."
      export FFTW="$FFTW_PATH"
      return 0
    fi

    if nm "$FFTW_PATH/lib/libfftw.a" 2>/dev/null | grep -Eq '(^|[[:space:]])_?fftw_create_plan([[:space:]]|$)'; then
      print_success "FFTW already built (double precision) at: $FFTW_PATH"
      export FFTW="$FFTW_PATH"
      return 0
    fi

    print_error "FFTW at $FFTW_PATH is not verified as double precision."
    print_info "Move or remove that directory yourself, then rerun this script to rebuild it."
    return 1
  fi

  if [ -d "$FFTW_PATH" ]; then
    print_info "Found FFTW source at $FFTW_PATH; build outputs are missing, so it will be built now."
  fi

  if [ ! -s "$FFTW_ARCHIVE" ]; then
    print_info "Downloading FFTW ${FFTW_VERSION}..."
    if ! curl -fSsL --retry 3 -o "$FFTW_ARCHIVE" "https://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz"; then
      rm -f "$FFTW_ARCHIVE"
      print_error "Could not download FFTW. Check your network connection and try again."
      return 1
    fi
  fi

  if ! tar -tzf "$FFTW_ARCHIVE" >/dev/null; then
    print_error "The FFTW archive is invalid or incomplete: $FFTW_ARCHIVE"
    print_info "Remove or replace that archive, then rerun this script."
    return 1
  fi

  if [ ! -d "$FFTW_PATH" ]; then
    print_info "Extracting FFTW..."
    if ! tar -xzf "$FFTW_ARCHIVE" -C "$UTILS_DIR"; then
      print_error "Could not extract FFTW into $UTILS_DIR"
      return 1
    fi
  fi

  if [ ! -f "$FFTW_PATH/configure" ]; then
    print_error "FFTW source directory is incomplete: $FFTW_PATH"
    print_info "Move or remove it yourself, then rerun this script to extract a clean copy."
    return 1
  fi

  # Update old config scripts bundled with FFTW 2.1.5 for modern build hosts.
  # Pin the GCC mirror revision so the downloaded scripts are reproducible.
  CONFIG_COMMIT="bec1b0a2f0c64ee462f6ba4f093803d0308d5a82"
  CONFIG_URL="https://raw.githubusercontent.com/gcc-mirror/gcc/${CONFIG_COMMIT}"
  if [ ! -f "$FFTW_PATH/config-scripts-downloaded" ] || [ "$(cat "$FFTW_PATH/config-scripts-downloaded" 2>/dev/null)" != "$CONFIG_COMMIT" ]; then
    print_info "Downloading config.guess and config.sub (commit $CONFIG_COMMIT)..."
    if ! curl -fSsL -o "$FFTW_PATH/config.guess.tmp" "$CONFIG_URL/config.guess" ||
      ! curl -fSsL -o "$FFTW_PATH/config.sub.tmp" "$CONFIG_URL/config.sub"; then
      rm -f "$FFTW_PATH/config.guess.tmp" "$FFTW_PATH/config.sub.tmp"
      print_error "Could not download the FFTW config scripts. Check your network connection and try again."
      return 1
    fi
    mv "$FFTW_PATH/config.guess.tmp" "$FFTW_PATH/config.guess"
    mv "$FFTW_PATH/config.sub.tmp" "$FFTW_PATH/config.sub"
    chmod +x "$FFTW_PATH/config.guess" "$FFTW_PATH/config.sub"
    printf '%s\n' "$CONFIG_COMMIT" > "$FFTW_PATH/config-scripts-downloaded"
  fi

  print_info "Configuring FFTW ..."
  if ! (cd "$FFTW_PATH" && set -x && ./configure --prefix="$FFTW_PATH" --disable-fortran CFLAGS="-fPIC"); then
    print_error "FFTW configuration failed."
    return 1
  fi
  print_info "Building FFTW (this may take several minutes)..."
  if ! (cd "$FFTW_PATH" && set -x && make_with_shell -j"$(nproc_opt)" && make_with_shell install); then
    print_error "FFTW build failed."
    return 1
  fi
  export FFTW="$FFTW_PATH"
  print_success "FFTW built at: $FFTW"
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

  print_info "Building CurveLab at: $FDCT"

  # Ensure FFTW directory layout matches what CurveLab's makefile.opt expects:
  #   ${FFTW_DIR}/fftw/fftw.h  and  ${FFTW_DIR}/fftw/.libs/libfftw.a
  mkdir -p "${FFTW}/fftw/.libs"
  # CurveLab's makefile expects headers and libraries under these paths.
  if [ -f "${FFTW}/include/fftw.h" ] && [ ! -f "${FFTW}/fftw/fftw.h" ]; then
    ln -sf "${FFTW}/include/fftw.h" "${FFTW}/fftw/fftw.h"
    print_info "Symlinked fftw.h → ${FFTW}/fftw/fftw.h"
  fi
  for lib in "${FFTW}/lib"/libfftw*; do
    [ -f "$lib" ] || continue
    base="$(basename "$lib")"
    if [ ! -f "${FFTW}/fftw/.libs/${base}" ]; then
      ln -sf "$lib" "${FFTW}/fftw/.libs/${base}"
      print_info "Symlinked ${base} → ${FFTW}/fftw/.libs/${base}"
    fi
  done

  # Clean and rebuild CurveLab components to ensure they link the correct FFTW
  for component in fdct_wrapping_cpp fdct fdct3d; do
    source_dir="$FDCT/$component/src"
    if [ -d "$source_dir" ]; then
      print_info "Cleaning and building CurveLab $component..."
      if ! (cd "$source_dir" && make_with_shell clean && make_with_shell "FFTW_DIR=$FFTW"); then
        print_error "CurveLab $component build failed."
        return 1
      fi
    fi
  done

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

  cd "$PROJECT_ROOT"

  # uv builds curvelops in an isolated environment and does not forward shell
  # exports. curvelops' setup.py reads FFTW and FDCT from that environment.
  # uv.toml is gitignored because the paths are local to this machine.
  print_info "Writing uv.toml so the curvelops build can see FFTW and CurveLab..."
  cat > "$PROJECT_ROOT/uv.toml" <<EOF
# Written by bin/install.sh. Local paths; do not commit.
extra-build-variables = { curvelops = { FFTW = "${FFTW}", FDCT = "${FDCT}" } }
EOF

  # Reinstall curvelops against the current FFTW/CurveLab without deleting the
  # user's entire environment or unrelated uv source-build caches.
  if ! uv cache clean curvelops; then
    print_warning "Could not clear the curvelops uv cache; continuing with a forced reinstall."
  fi

  print_info "Syncing uv environment..."
  if ! uv sync --python "$PYTHON_VERSION" --extra curvelops --extra segmentation --extra fiji --reinstall-package curvelops; then
    print_error "uv sync failed."
    return 1
  fi
  print_success "Environment configured."

  print_info "Installing tme-quant..."
  if ! uv pip install -e .; then
    print_error "Installing tme-quant failed."
    return 1
  fi

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

  if uv run python -c "
import sys
errors = []
try:
    import pycurvelets as pc
    print('  ✓ pycurvelets')
    print(f'    HAS_CURVELETS = {pc.HAS_CURVELETS}')
    if not pc.HAS_CURVELETS:
        errors.append('pycurvelets: HAS_CURVELETS is False (curvelet backend not functional; check FFTW/FDCT)')
except Exception as e:
    errors.append(f'pycurvelets: {e}')
try:
    import curvelops
    print(f'  ✓ curvelops {curvelops.__version__}')
except Exception as e:
    errors.append(f'curvelops: {e}')
try:
    import napari_curvealign
    print('  ✓ napari_curvealign')
except Exception as e:
    errors.append(f'napari_curvealign: {e}')
try:
    import appose
    print('  ✓ appose-python worker')
except Exception as e:
    errors.append(f'appose-python worker: {e}')
if errors:
    print('\\n✗ Validation failed:')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)
print('\\n✅ All imports successful!')
"; then
    print_success "Verification passed."
  else
    print_error "Verification failed."
    return 1
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
  echo "  1. Run this script from a checked-out copy of the repository"
  echo "  2. Install uv (https://docs.astral.sh/uv/)"
  echo "  3. Download CurveLab into $UTILS_DIR (see https://curvelet.org/download.php)"
  echo "     Expected folder name: CurveLab-<version>"
  echo ""
  echo "This script will:"
  echo "  - Download and build FFTW in $UTILS_DIR"
  echo "  - Build CurveLab from $UTILS_DIR"
  echo "  - Create and sync uv env with needed packages"
  echo "  - Run verification to ensure all is functional"
  echo ""
  read -p "Press Enter to continue or Ctrl+C to cancel..."
  echo ""

  ensure_uv
  mkdir -p "$UTILS_DIR"
  ensure_build_tools
  ensure_native_python_compiler
  check_curvelab
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
  echo "Plugins → napari-curvealign (or CurveAlign for Napari)"
  echo ""
}

main "$@"
