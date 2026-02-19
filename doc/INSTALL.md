# Installation

## Prerequisites

1. **Clone this repository**
   ```bash
   git clone https://github.com/uw-loci/tme-quant.git
   cd tme-quant
   ```

2. **Install uv** (https://docs.astral.sh/uv/)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Download CurveLab** (required; cannot be redistributed)
   - https://curvelet.org/download.php
   - Extract to `../utils/` relative to tme-quant (e.g. `../utils/CurveLab-2.1.3`)

## Install

```bash
bash bin/install.sh
```

The script checks for uv, downloads FFTW, detects CurveLab in `../utils`, builds both, syncs the env, and verifies.

## Run

```bash
uv run napari
```

**Plugins → napari-curvealign** (or **CurveAlign for Napari**)

## Directory layout

```
parent/
├── tme-quant/          # This repo
└── utils/
    ├── fftw-2.1.5/     # Created by install.sh
    └── CurveLab-2.1.3/ # You must download and extract here
```

## Without curvelops

To run napari without the curvelet backend (no FFTW/CurveLab required):

```bash
uv sync
uv run napari
```

**Note:** The napari plugin will load, but curvelet-based analysis (e.g. fiber orientation) will not run. You get the UI with mock/placeholder results. For real curvelet analysis, use the full install above.
