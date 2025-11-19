#!/bin/bash
################################################################################
# tme-quant Installation Script
# 
# This script automates the complete installation of the CurveAlign Napari plugin
# including all dependencies and Curvelops with FFTW/CurveLab support.
#
# Prerequisites:
#   - Python 3.9+ installed
#   - FFTW 2.1.5 installed in ../utils/fftw-2.1.5 (optional, for curvelet backend)
#   - CurveLab 2.1.2 installed in ../utils/CurveLab-2.1.2 (optional, for curvelet backend)
#
# Usage:
#   bash bin/install.sh
#   or
#   chmod +x bin/install.sh && ./bin/install.sh
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script and project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
UTILS_DIR="$PROJECT_ROOT/../utils"

################################################################################
# Helper Functions
################################################################################

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_header() {
    echo -e "${CYAN}$1${NC}"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

################################################################################
# Step 1: Check Prerequisites
################################################################################

check_prerequisites() {
    print_header "═══════════════════════════════════════════════════════════════"
    print_header "  Checking Prerequisites"
    print_header "═══════════════════════════════════════════════════════════════"
    echo ""
    
    # Check Python
    if check_command python3; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python found: $PYTHON_VERSION"
        
        # Check Python version (3.9+)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
            print_error "Python 3.9+ required. Found: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.9 or later."
        exit 1
    fi
    
    # Check pip
    if check_command pip3 || python3 -m pip --version &> /dev/null; then
        print_success "pip found"
    else
        print_error "pip not found. Please install pip."
        exit 1
    fi
    
    # Resolve UTILS_DIR to absolute path
    UTILS_DIR_ABS="$(cd "$UTILS_DIR" && pwd 2>/dev/null || echo "$UTILS_DIR")"
    
    # Check FFTW
    FFTW_PATH="$UTILS_DIR_ABS/fftw-2.1.5"
    if [ -d "$FFTW_PATH" ]; then
        print_success "FFTW found at: $FFTW_PATH"
        export FFTW="$FFTW_PATH"
    else
        print_error "FFTW not found at: $FFTW_PATH"
        echo "   Expected location: $UTILS_DIR_ABS/fftw-2.1.5"
        echo "   Please ensure FFTW 2.1.5 is installed in the utils/ directory (parallel to tme-quant/)."
        exit 1
    fi
    
    # Check CurveLab
    CURVELAB_PATH="$UTILS_DIR_ABS/CurveLab-2.1.2"
    if [ -d "$CURVELAB_PATH" ]; then
        print_success "CurveLab found at: $CURVELAB_PATH"
        export FDCT="$CURVELAB_PATH"
    else
        print_error "CurveLab not found at: $CURVELAB_PATH"
        echo "   Expected location: $UTILS_DIR_ABS/CurveLab-2.1.2"
        echo "   Please ensure CurveLab 2.1.2 is installed in the utils/ directory (parallel to tme-quant/)."
        exit 1
    fi
    
    echo ""
}

################################################################################
# Step 2: Setup Virtual Environment
################################################################################

setup_venv() {
    print_header "═══════════════════════════════════════════════════════════════"
    print_header "  Setting Up Virtual Environment"
    print_header "═══════════════════════════════════════════════════════════════"
    echo ""
    
    cd "$PROJECT_ROOT"
    
    if [ -d ".venv" ]; then
        print_warning "Virtual environment already exists at .venv"
        read -p "Remove existing virtual environment and create new one? [y/N]: " recreate
        if [[ "$recreate" =~ ^[Yy]$ ]]; then
            print_info "Removing existing virtual environment..."
            rm -rf .venv
        else
            print_info "Using existing virtual environment"
            return 0
        fi
    fi
    
    print_info "Creating virtual environment..."
    python3 -m venv .venv
    
    print_success "Virtual environment created"
    echo ""
}

################################################################################
# Step 3: Activate Environment and Setup Curvelops
################################################################################

activate_and_setup() {
    print_header "═══════════════════════════════════════════════════════════════"
    print_header "  Activating Environment and Configuring Curvelops"
    print_header "═══════════════════════════════════════════════════════════════"
    echo ""
    
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment
    print_info "Activating virtual environment..."
    source .venv/bin/activate
    
    # Upgrade pip, setuptools, wheel
    print_info "Upgrading pip, setuptools, and wheel..."
    pip install --upgrade pip setuptools wheel --quiet
    
    # Source the curvelops environment setup script
    print_info "Configuring Curvelops environment variables..."
    source "$SCRIPT_DIR/setup_curvelops_env.sh"
    
    # Verify environment variables are set
    if [ -z "$FFTW" ] || [ -z "$FDCT" ]; then
        print_error "Failed to set FFTW and FDCT environment variables"
        exit 1
    fi
    
    print_success "Environment configured"
    echo "  FFTW: $FFTW"
    echo "  FDCT: $FDCT"
    echo ""
}

################################################################################
# Step 4: Install Dependencies
################################################################################

install_dependencies() {
    print_header "═══════════════════════════════════════════════════════════════"
    print_header "  Installing Dependencies"
    print_header "═══════════════════════════════════════════════════════════════"
    echo ""
    
    cd "$PROJECT_ROOT"
    
    # Ensure we're in the virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        source .venv/bin/activate
        source "$SCRIPT_DIR/setup_curvelops_env.sh"
    fi
    
    # Install curvelops first (requires FFTW/FDCT env vars)
    print_info "Installing Curvelops (this may take a few minutes)..."
    if pip install "curvelops @ git+https://github.com/PyLops/curvelops@0.23" --quiet; then
        print_success "Curvelops installed successfully"
    else
        print_error "Failed to install Curvelops"
        print_warning "Make sure FFTW and CurveLab paths are correct"
        exit 1
    fi
    echo ""
    
    # Install the main package with all dependencies
    print_info "Installing tme-quant and all dependencies..."
    print_info "This will install:"
    echo "  - Core dependencies (numpy, scipy, pandas, etc.)"
    echo "  - Napari and all GUI dependencies"
    echo "  - ROI management tools"
    echo "  - Image processing libraries"
    echo ""
    
    if pip install -e . --quiet; then
        print_success "All dependencies installed successfully"
    else
        print_error "Failed to install package dependencies"
        exit 1
    fi
    echo ""
}

################################################################################
# Step 5: Verify Installation
################################################################################

verify_installation() {
    print_header "═══════════════════════════════════════════════════════════════"
    print_header "  Verifying Installation"
    print_header "═══════════════════════════════════════════════════════════════"
    echo ""
    
    cd "$PROJECT_ROOT"
    
    # Ensure we're in the virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        source .venv/bin/activate
    fi
    
    print_info "Testing imports..."
    
    python3 << 'EOF'
import sys
errors = []

try:
    import napari
    print(f"  ✓ napari {napari.__version__}")
except ImportError as e:
    errors.append(f"napari: {e}")

try:
    import napari_curvealign
    print("  ✓ napari_curvealign")
except ImportError as e:
    errors.append(f"napari_curvealign: {e}")

try:
    import curvelops
    print(f"  ✓ curvelops {curvelops.__version__}")
except ImportError as e:
    errors.append(f"curvelops: {e}")

try:
    import roifile
    print(f"  ✓ roifile {roifile.__version__}")
except ImportError as e:
    errors.append(f"roifile: {e}")

try:
    import numpy
    print(f"  ✓ numpy {numpy.__version__}")
except ImportError as e:
    errors.append(f"numpy: {e}")

try:
    import pandas
    print(f"  ✓ pandas {pandas.__version__}")
except ImportError as e:
    errors.append(f"pandas: {e}")

if errors:
    print("\n✗ Import errors:")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)
else:
    print("\n✅ All imports successful!")
EOF

    if [ $? -eq 0 ]; then
        print_success "Installation verified successfully"
        return 0
    else
        print_error "Verification failed"
        return 1
    fi
}

################################################################################
# Step 6: Create Activation Helper
################################################################################

create_activation_helper() {
    print_header "═══════════════════════════════════════════════════════════════"
    print_header "  Creating Activation Helper"
    print_header "═══════════════════════════════════════════════════════════════"
    echo ""
    
    # Create activation script in bin/ directory
    cat > "$SCRIPT_DIR/activate.sh" << 'ACTIVATE_EOF'
#!/bin/bash
# Quick activation script for tme-quant
# Usage: source bin/activate.sh
#
# This script activates the virtual environment and sets up Curvelops
# environment variables for FFTW and CurveLab.

BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$BIN_DIR")"

# Check if virtual environment exists
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run: bash bin/install.sh"
    return 1 2>/dev/null || exit 1
fi

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Set up Curvelops environment
if [ -f "$BIN_DIR/setup_curvelops_env.sh" ]; then
    source "$BIN_DIR/setup_curvelops_env.sh"
else
    echo "⚠️  Warning: setup_curvelops_env.sh not found"
fi

echo "✅ Environment activated!"
echo "   Project: $PROJECT_ROOT"
echo "   Python: $(python --version 2>/dev/null || echo 'unknown')"
if [ -n "$FFTW" ] && [ -n "$FDCT" ]; then
    echo "   FFTW: $FFTW"
    echo "   CurveLab: $FDCT"
fi
ACTIVATE_EOF

    chmod +x "$SCRIPT_DIR/activate.sh"
    print_success "Created activation script: bin/activate.sh"
    echo ""
}

################################################################################
# Main Installation Flow
################################################################################

main() {
    clear
    echo ""
    print_header "═══════════════════════════════════════════════════════════════"
    print_header "  tme-quant Installation"
    print_header "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "This script will:"
    echo "  1. Check prerequisites (Python, FFTW, CurveLab)"
    echo "  2. Create a virtual environment"
    echo "  3. Install all dependencies including Curvelops"
    echo "  4. Verify the installation"
    echo ""
    read -p "Press Enter to continue or Ctrl+C to cancel..."
    echo ""
    
    check_prerequisites
    setup_venv
    activate_and_setup
    install_dependencies
    
    if verify_installation; then
        create_activation_helper
        echo ""
        print_header "═══════════════════════════════════════════════════════════════"
        print_success "Installation Complete!"
        print_header "═══════════════════════════════════════════════════════════════"
        echo ""
        echo "To use the plugin:"
        echo ""
        echo "  1. Activate the environment:"
        echo "     source bin/activate.sh"
        echo "     (or: source .venv/bin/activate && source bin/setup_curvelops_env.sh)"
        echo ""
        echo "  2. Launch Napari:"
        echo "     napari"
        echo ""
        echo "  3. The CurveAlign widget will appear in:"
        echo "     Plugins → napari-curvealign → CurveAlign Widget"
        echo ""
        echo "For more information, see README.md"
        echo ""
    else
        print_error "Installation completed but verification failed"
        echo "Please check the error messages above and try again."
        exit 1
    fi
}

# Run main
main "$@"

