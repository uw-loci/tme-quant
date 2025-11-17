#!/bin/bash
################################################################################
# tme-quant Installation Script
# This script automates the installation of tme-quant and its dependencies
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script and project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
ENV_NAME="tme-quant"
PYTHON_VERSION="3.11"
FFTW_VERSION="2.1.5"
CURVELAB_VERSION="2.1.3"  # Update this when you have a specific version

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

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

################################################################################
# Step 1: Check and Install System Dependencies
################################################################################

check_system_deps() {
    print_info "Checking system dependencies..."
    
    # Check for conda/miniconda/anaconda
    if check_command conda; then
        print_success "Conda found: $(conda --version)"
    else
        print_warning "Conda not found. Attempting to install miniconda..."
        install_miniconda
    fi
    
    # Check for build tools
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if ! check_command gcc; then
            print_warning "Xcode Command Line Tools required"
            print_info "Installing Xcode Command Line Tools..."
            xcode-select --install 2>/dev/null || true
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if ! check_command gcc; then
            print_warning "Install build-essential with: sudo apt-get install build-essential"
        fi
    fi
}

install_miniconda() {
    local install_dir="$HOME/miniconda"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        local installer="Miniconda3-latest-MacOSX-x86_64.sh"
        local arch="arm64"
        if [[ $(uname -m) == "arm64" ]] || [[ $(uname -m) == "aarch64" ]]; then
            installer="Miniconda3-latest-MacOSX-arm64.sh"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        installer="Miniconda3-latest-Linux-x86_64.sh"
    else
        print_error "Unsupported OS. Please install conda manually."
        exit 1
    fi
    
    print_info "Downloading Miniconda..."
    cd /tmp
    curl -O "https://repo.anaconda.com/miniconda/$installer"
    
    print_info "Installing Miniconda to $install_dir..."
    bash "$installer" -b -p "$install_dir"
    
    # Initialize conda
    "$install_dir/bin/conda" init bash
    source "$HOME/.bashrc" 2>/dev/null || true
    
    print_success "Miniconda installed"
    export PATH="$install_dir/bin:$PATH"
}

################################################################################
# Step 2: Setup Python Environment
################################################################################

setup_python_env() {
    print_info "Setting up Python environment..."
    
    # Initialize conda in this shell
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    fi
    
    # Create conda environment if it doesn't exist
    if conda env list | grep -q "^$ENV_NAME "; then
        print_warning "Environment '$ENV_NAME' already exists. Skipping creation."
    else
        print_info "Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
        conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION" pip
    fi
    
    # Activate environment
    print_info "Activating environment: $ENV_NAME"
    conda activate "$ENV_NAME"
    
    print_success "Python environment ready"
}

################################################################################
# Step 3: Install CurveLab (Manual download required)
################################################################################

setup_curvelab() {
    print_info "Checking for CurveLab installation..."
    
    # Check if CurveLab is already available
    if [ -n "${FDCT:-}" ] && [ -d "$FDCT" ]; then
        print_success "CurveLab found at: $FDCT"
        return 0
    fi
    
    # Auto-detect CurveLab under ../utils
    local utils_fdct
    for d in "$PROJECT_ROOT/../utils"/CurveLab-*; do
        if [ -d "$d" ]; then
            utils_fdct="$d"
            break
        fi
    done
    if [ -z "${FDCT:-}" ] && [ -n "$utils_fdct" ]; then
        export FDCT="$utils_fdct"
        print_success "Auto-detected CurveLab at: $FDCT"
        return 0
    fi
    
    # Prompt user for CurveLab location
    print_warning "CurveLab not found in environment."
    echo ""
    echo "CurveLab is required for the curvelet backend (curvelops)."
    echo "You need to:"
    echo "  1. Agree to the CurveLab license: https://curvelet.org/"
    echo "  2. Download CurveLab from: https://curvelet.org/download.php"
    echo "  3. Extract it to a location on your system"
    echo ""
    
    read -p "Enter the path to your CurveLab installation (or press Enter to skip): " curvelab_path
    
    if [ -n "$curvelab_path" ] && [ -d "$curvelab_path" ]; then
        export FDCT="$curvelab_path"
        print_success "CurveLab path set to: $FDCT"
        
        # Check if CurveLab needs to be built
        if [ ! -f "$FDCT/fdct_wrapping_cpp/src/fdct_wrapping_cpp" ]; then
            print_info "Building CurveLab components..."
            cd "$FDCT"
            
            if [ -d "fdct_wrapping_cpp/src" ]; then
                (cd fdct_wrapping_cpp/src && make) || print_warning "fdct_wrapping_cpp build failed"
            fi
            if [ -d "fdct3d/src" ]; then
                (cd fdct3d/src && make) || print_warning "fdct3d build failed"
            fi
            
            cd "$PROJECT_ROOT"
        fi
    else
        print_warning "Skipping CurveLab setup. You can install it later."
    fi
}

################################################################################
# Step 4: Download and Build FFTW
################################################################################

setup_fftw() {
    print_info "Checking for FFTW installation..."
    
    # Check if FFTW is already available
    if [ -n "${FFTW:-}" ] && [ -d "$FFTW" ]; then
        print_success "FFTW found at: $FFTW"
        return 0
    fi
    
    # Check common locations
    local fftw_locations=(
        "$PROJECT_ROOT/../utils/fftw-2.1.5"
        "$HOME/opt/fftw-2.1.5"
        "/usr/local/fftw-2.1.5"
        "/opt/fftw-2.1.5"
    )
    
    for loc in "${fftw_locations[@]}"; do
        if [ -d "$loc" ]; then
            export FFTW="$loc"
            print_success "FFTW found at: $FFTW"
            return 0
        fi
    done
    
    print_info "FFTW not found. Building FFTW $FFTW_VERSION..."
    
    # Ask user if they want to build FFTW
    echo ""
    read -p "Build FFTW automatically? This will download and compile FFTW (10-15 min) [y/N]: " build_fftw
    
    if [[ ! "$build_fftw" =~ ^[Yy]$ ]]; then
        print_warning "Skipping FFTW setup. You can install it later."
        return 0
    fi
    
    local utils_dir="$PROJECT_ROOT/../utils"
    mkdir -p "$utils_dir"
    cd "$utils_dir"
    
    # Download FFTW
    if [ ! -f "fftw-$FFTW_VERSION.tar.gz" ]; then
        print_info "Downloading FFTW..."
        curl -L -O "http://www.fftw.org/fftw-$FFTW_VERSION.tar.gz"
    fi
    
    if [ ! -d "fftw-$FFTW_VERSION" ]; then
        print_info "Extracting FFTW..."
        tar xzf "fftw-$FFTW_VERSION.tar.gz"
    fi
    
    cd "fftw-$FFTW_VERSION"
    
    print_info "Configuring FFTW..."
    ./configure --prefix="$(pwd)" --disable-fortran
    
    print_info "Building FFTW (this may take 10-15 minutes)..."
    make -j"$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"
    
    print_info "Installing FFTW..."
    make install
    
    # Create compatibility symlink
    ln -sf include fftw 2>/dev/null || true
    
    export FFTW="$(pwd)"
    print_success "FFTW built and installed at: $FFTW"
    
    cd "$PROJECT_ROOT"
}

################################################################################
# Step 5: Install tme-quant Package
################################################################################

install_package() {
    print_info "Installing tme-quant package..."
    
    cd "$PROJECT_ROOT"
    
    # Update pip
    pip install --upgrade pip
    
    # Install base package
    print_info "Installing base dependencies..."
    pip install -e .
    
    # Check if CurveLab is available for curvelops
    if [ -n "${FDCT:-}" ] && [ -n "${FFTW:-}" ]; then
        print_info "CurveLab and FFTW detected. Installing curvelops option..."
        
        # Set environment variables for curvelops build
        export CPPFLAGS="-I${FFTW}/include"
        export LDFLAGS="-L${FFTW}/lib"
        
        # Install with curvelops
        pip install -e ".[curvelops]" || print_warning "Failed to install curvelops option"
    else
        print_info "Installing without curvelops (basic mode)"
        pip install -e ".[dev]"
    fi
    
    print_success "tme-quant installed successfully"
}

################################################################################
# Step 6: Verify Installation
################################################################################

verify_installation() {
    print_info "Verifying installation..."
    
    cd "$PROJECT_ROOT"
    
    # Test Python imports
    python << 'EOF'
try:
    import curvealign_py as curvealign
    import ctfire_py as ctfire
    print("\n✅ Core packages imported successfully")
    
    # Check curvelops status
    from curvealign_py.core.algorithms.fdct_wrapper import get_curvelops_status
    status = get_curvelops_status()
    
    if status['available']:
        print(f"✅ Curvelops available: version {status.get('version', 'unknown')}")
    else:
        print("ℹ️  Curvelops not available (using placeholder mode)")
        
except ImportError as e:
    print(f"\n❌ Import failed: {e}")
    exit(1)
EOF
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_success "Installation verified"
    else
        print_error "Verification failed"
        exit 1
    fi
}

################################################################################
# Step 7: Create convenience script
################################################################################

create_activation_script() {
    print_info "Creating activation script..."
    
    cat > "$PROJECT_ROOT/bin/activate_env.sh" << 'EOF'
#!/bin/bash
# Activate tme-quant environment and set up paths

# Initialize conda
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

# Activate environment
conda activate tme-quant

# Set environment variables if needed
if [ -n "$FDCT" ] && [ -d "$FDCT" ]; then
    export FDCT
fi
if [ -n "$FFTW" ] && [ -d "$FFTW" ]; then
    export FFTW
    export CPPFLAGS="-I${FFTW}/include"
    export LDFLAGS="-L${FFTW}/lib"
fi

echo "tme-quant environment activated"
EOF
    
    chmod +x "$PROJECT_ROOT/bin/activate_env.sh"
    print_success "Created activation script: bin/activate_env.sh"
}

################################################################################
# Main Installation Flow
################################################################################

main() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  tme-quant Installation Script"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    
    check_system_deps
    echo ""
    
    setup_python_env
    echo ""
    
    setup_fftw
    echo ""
    
    setup_curvelab
    echo ""
    
    install_package
    echo ""
    
    verify_installation
    echo ""
    
    create_activation_script
    echo ""
    
    echo "═══════════════════════════════════════════════════════════════"
    print_success "Installation Complete!"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "To use tme-quant:"
    echo "  1. Activate the environment: source bin/activate_env.sh"
    echo "  2. Or manually: conda activate $ENV_NAME"
    echo ""
    echo "Optional: Run tests with:"
    echo "  pytest tests/ -v"
    echo ""
}

# Run main
main "$@"

