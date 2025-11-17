# Napari Plugin Installation Guide

Complete guide for installing the CurveAlign Napari plugin with all dependencies including Curvelops support.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [One-Click Installation](#one-click-installation)
3. [Manual Installation](#manual-installation)
4. [Verification](#verification)
5. [Usage](#usage)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before installing, ensure you have:

### Required
- **Python 3.9+** installed and accessible via `python3`
- **FFTW 2.1.5** installed in `../utils/fftw-2.1.5`
- **CurveLab 2.1.2** installed in `../utils/CurveLab-2.1.2`

### Directory Structure

The installation script expects the following directory structure:

```
Parent_dir/
├── utils/
│   ├── fftw-2.1.5/          # FFTW installation
│   └── CurveLab-2.1.2/      # CurveLab installation
└── tme-quant-napari-curvealign/
    ├── bin/
    │   ├── install.sh        # Main installation script
    │   ├── setup_curvelops_env.sh
    │   └── activate.sh       # Generated after installation
    ├── src/
    └── pyproject.toml
```

### Verifying Prerequisites

Check that FFTW and CurveLab are in the correct locations:

```bash
# Check FFTW
ls -d ../utils/fftw-2.1.5
# Should show: ../utils/fftw-2.1.5

# Check CurveLab
ls -d ../utils/CurveLab-2.1.2
# Should show: ../utils/CurveLab-2.1.2
```

---

## One-Click Installation

The easiest way to install everything is using the automated installation script.

### Step 1: Navigate to Project Directory

```bash
cd /path/to/tme-quant-napari-curvealign
```

### Step 2: Run Installation Script

```bash
bash bin/install.sh
```

### What the Script Does

The installation script automatically:

1. ✅ **Checks Prerequisites**
   - Verifies Python 3.9+ is installed
   - Checks for pip
   - Validates FFTW and CurveLab locations

2. ✅ **Creates Virtual Environment**
   - Creates `.venv/` directory
   - Sets up isolated Python environment

3. ✅ **Configures Curvelops**
   - Sources `bin/setup_curvelops_env.sh`
   - Sets `FFTW` and `FDCT` environment variables
   - Configures compiler flags (`CPPFLAGS`, `LDFLAGS`)

4. ✅ **Installs Dependencies**
   - Upgrades pip, setuptools, wheel
   - Installs Curvelops with FFTW/CurveLab support
   - Installs all napari dependencies
   - Installs core scientific libraries
   - Installs ROI management tools

5. ✅ **Verifies Installation**
   - Tests imports of key packages
   - Confirms all dependencies are working

6. ✅ **Creates Activation Script**
   - Generates `bin/activate.sh` for easy environment activation

### Installation Output

You'll see progress messages like:

```
═══════════════════════════════════════════════════════════════
  Checking Prerequisites
═══════════════════════════════════════════════════════════════

[✓] Python found: 3.13.3
[✓] pip found
[✓] FFTW found at: /path/to/utils/fftw-2.1.5
[✓] CurveLab found at: /path/to/utils/CurveLab-2.1.2

═══════════════════════════════════════════════════════════════
  Setting Up Virtual Environment
═══════════════════════════════════════════════════════════════

[INFO] Creating virtual environment...
[✓] Virtual environment created

═══════════════════════════════════════════════════════════════
  Installing Dependencies
═══════════════════════════════════════════════════════════════

[INFO] Installing Curvelops (this may take a few minutes)...
[✓] Curvelops installed successfully

[INFO] Installing tme-quant-napari-curvealign and all dependencies...
[✓] All dependencies installed successfully

═══════════════════════════════════════════════════════════════
  Verifying Installation
═══════════════════════════════════════════════════════════════

  ✓ napari 0.6.6
  ✓ napari_curvealign
  ✓ curvelops 0.23
  ✓ roifile 2025.5.10
  ✓ numpy 2.2.6
  ✓ pandas 2.3.3

✅ All imports successful!
[✓] Installation verified successfully
```

---

## Manual Installation

If you prefer to install semi-automated or the automated script fails, follow these steps:

### Step 1: Create Virtual Environment

```bash
cd tme-quant-napari-curvealign
python3 -m venv .venv
```

### Step 2: Activate Virtual Environment

```bash
source .venv/bin/activate
```

### Step 3: Upgrade Build Tools

```bash
pip install --upgrade pip setuptools wheel
```

### Step 4: Set Up Curvelops Environment

```bash
source bin/setup_curvelops_env.sh
```

This sets:
- `FFTW=/path/to/utils/fftw-2.1.5`
- `FDCT=/path/to/utils/CurveLab-2.1.2`
- `CPPFLAGS=-I${FFTW}/include`
- `LDFLAGS=-L${FFTW}/lib`

### Step 5: Install Curvelops

```bash
pip install "curvelops @ git+https://github.com/PyLops/curvelops@0.23"
```

### Step 6: Install Package

```bash
pip install -e .
```

This installs all dependencies from `pyproject.toml`:
- Core dependencies (numpy, scipy, pandas, etc.)
- Napari and GUI dependencies
- ROI management tools
- Image processing libraries

---

## Verification

After installation, verify everything works:

### Quick Verification

```bash
source bin/activate.sh
python -c "import napari; import napari_curvealign; import curvelops; print('✅ All imports successful!')"
```

### Detailed Verification

```bash
source bin/activate.sh
python << EOF
import napari
import napari_curvealign
import curvelops
import roifile
import numpy
import pandas

print(f"✅ napari {napari.__version__}")
print(f"✅ curvelops {curvelops.__version__}")
print(f"✅ roifile {roifile.__version__}")
print(f"✅ numpy {numpy.__version__}")
print(f"✅ pandas {pandas.__version__}")
print("\n✅ All packages imported successfully!")
EOF
```

---

## Usage

### Activating the Environment

After installation, activate the environment:

```bash
source bin/activate.sh
```

This will:
- Activate the virtual environment
- Set up Curvelops environment variables
- Display status information

### Launching Napari

```bash
napari
```

### Accessing the Plugin

Once Napari is open:

1. Go to **Plugins** menu
2. Select **napari-curvealign**
3. Click **CurveAlign Widget**

The widget will appear as a dockable panel in Napari.

### Plugin Features

The CurveAlign widget provides:

- **Main Tab**: Image loading, analysis parameters, run analysis
- **Preprocessing Tab**: Tubeness, Frangi filters, thresholding
- **Segmentation Tab**: Automated ROI generation (Cellpose, StarDist, Threshold)
- **ROI Manager Tab**: Manual ROI creation, management, and analysis

---

## Troubleshooting

### Installation Issues

#### FFTW/CurveLab Not Found

**Error:** `FFTW not found at: ../utils/fftw-2.1.5`

**Solution:**
1. Verify FFTW is installed in the correct location
2. Check the path: `ls -d ../utils/fftw-2.1.5`
3. If using a different path, manually set environment variables:
   ```bash
   export FFTW="/path/to/your/fftw-2.1.5"
   export FDCT="/path/to/your/CurveLab-2.1.2"
   ```

#### Curvelops Build Fails

**Error:** `Failed to build 'curvelops'`

**Solution:**
1. Ensure FFTW and CurveLab are correctly installed
2. Verify environment variables are set:
   ```bash
   echo $FFTW
   echo $FDCT
   echo $CPPFLAGS
   echo $LDFLAGS
   ```
3. Check that FFTW includes and libs exist:
   ```bash
   ls $FFTW/include
   ls $FFTW/lib
   ```

#### Python Version Issues

**Error:** `Python 3.9+ required. Found: 3.8.x`

**Solution:**
- Install Python 3.9 or later
- On macOS: `brew install python@3.11`
- On Linux: Use your package manager or download from python.org

### Runtime Issues

#### Plugin Not Appearing in Napari

**Solution:**
1. Verify installation:
   ```bash
   source bin/activate.sh
   python -c "import napari_curvealign; print('Plugin installed')"
   ```
2. Check Napari plugin discovery:
   ```bash
   napari --info
   ```
3. Reinstall in development mode:
   ```bash
   pip install -e .
   ```

#### Import Errors

**Error:** `ModuleNotFoundError: No module named 'napari'`

**Solution:**
1. Ensure virtual environment is activated
2. Reinstall dependencies:
   ```bash
   source bin/activate.sh
   pip install -e .
   ```

#### Curvelops Not Working

**Error:** Curvelops imports but fails at runtime

**Solution:**
1. Verify environment variables are set:
   ```bash
   source bin/activate.sh
   echo $FFTW
   echo $FDCT
   ```
2. Test Curvelops directly:
   ```bash
   python -c "from curvelops import FDCT2D; print('Curvelops OK')"
   ```

### Getting Help

If you encounter issues not covered here:

1. Check the main [README.md](../../README.md)
2. Review [TESTING_GUIDE.md](../../TESTING_GUIDE.md)
3. Check [SEGMENTATION_FEATURE.md](../../SEGMENTATION_FEATURE.md) for segmentation-specific issues
4. See [MULTI_ENVIRONMENT_GUIDE.md](../../MULTI_ENVIRONMENT_GUIDE.md) for multi-environment setups

---

## Scripts Reference

All installation and setup scripts are located in `bin/`:

| Script | Purpose |
|--------|---------|
| `bin/install.sh` | Main automated installation script |
| `bin/activate.sh` | Quick environment activation (generated by install.sh) |
| `bin/setup_curvelops_env.sh` | Sets up FFTW/CurveLab environment variables |
| `bin/setup.sh` | Original setup script (for base tme-quant) |
| `bin/activate_env.sh` | Original activation script (for base tme-quant) |

---

## Next Steps

After successful installation:

1. **Try the Examples:**
   ```bash
   source bin/activate.sh
   python examples/test_plugin.py
   ```

2. **Run Tests:**
   ```bash
   pytest tests/ -v
   ```

3. **Read the Documentation:**
   - [API Documentation](../curvealign/CURVEALIGN_PYTHON_API_SUMMARY.md)
   - [Architecture Overview](../curvealign/ARCHITECTURE.md)
   - [Testing Guide](../../TESTING_GUIDE.md)

4. **Explore Features:**
   - ROI Management: See `src/napari_curvealign/roi_manager.py`
   - Segmentation: See `SEGMENTATION_FEATURE.md`
   - Preprocessing: See `src/napari_curvealign/preprocessing.py`

---

**Last Updated:** 2024-11-12  
**Version:** 0.1.0

