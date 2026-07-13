# Development

Use the same install path as users: `bash bin/install.sh` (or `make setup`).

## Napari plugin

The plugin is registered via:
- `src/napari_curvealign/napari.yaml` – manifest (commands, widgets)
- `pyproject.toml` – entry point: `napari-curvealign = "napari_curvealign:napari.yaml"`

After `uv pip install -e .`, run `uv run napari` and open **Plugins → napari-curvealign**.

## Running tests

```bash
make test
```

Headless (no GUI): `QT_QPA_PLATFORM=offscreen make test`

Curvelet tests run automatically when curvelops is installed; otherwise they are skipped.

### Which environment (Windows / MSYS2)

The C++ `fiber_backend` extension is ABI-specific: pytest must run under the **same Python
that the `.pyd`/`.so` was built for**, or `CPP_AVAILABLE` is `False` and every backend test
silently *skips* instead of running. On Windows the correct environment is `.venv` at the
repo root, built on the **MSYS2 UCRT64 Python 3.14** (matching
`fiber_backend.cp314-mingw_x86_64_ucrt_gnu.pyd`):

| Property | Value |
|---|---|
| Location | `.venv/` (repo root) |
| Base Python | MSYS2 UCRT64 Python 3.14 (`C:/msys64/ucrt64/bin/python.exe`) |
| Created with | `python -m venv --system-site-packages` (inherits numpy/scipy/skimage/matplotlib) |
| Has | `pytest`, `curvelops 0.23.4`; `HAS_FIBER_BACKEND` and `HAS_CURVELOPS` both `True` |

Run the full suite from an MSYS2 UCRT64 login shell:

```bash
C:\msys64\usr\bin\bash.exe -lc "export MSYSTEM=UCRT64 && source /etc/profile && \
  cd /h/GitHub.06.2022/tmequant_ctfire/tme-quant && source .venv/bin/activate && \
  export TMEQ_RUN_CURVELETS=1 && export MPLCONFIGDIR=/c/msys64/tmp/mpl && \
  python -m pytest tests/ -v"
```

- `TMEQ_RUN_CURVELETS=1` is required to un-skip `tests/test_ct_fire.py` (curvelet path).
- `MPLCONFIGDIR=...` silences a matplotlib home-dir cache warning under the login shell.
- Point pytest at `tests/test_ct_fire.py` for just the 56 CT-FIRE tests, or `tests/` for all.

Do **not** run the suite with an unrelated interpreter (e.g. the Windows Store Python or a
throwaway `%TEMP%` venv) — the backend won't import and the C++-backed tests will skip
rather than fail, hiding regressions.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| uv not found | Install from https://docs.astral.sh/uv/ |
| FFTW build errors | macOS: `xcode-select --install`; Linux: `apt-get install build-essential gcc g++ make curl`; use `CFLAGS="-fPIC"` |
| CurveLab not found | Download from curvelet.org, place in `../utils/` |
| curvelops build errors | Ensure `FFTW` and `FDCT` (or `CPPFLAGS`/`LDFLAGS`) point to install roots |
| Plugin not showing | `uv pip install -e .`

## Development log

Important changes (`feat` / `fix` / `perf` / `refactor`) should be summarized here
while a draft pull request (or standard pull request) is open.

Workflow:
- Add or refine entries during PR preparation.
- Keep entries human-reviewed and editable until the PR is approved.
- Do not auto-commit development-note updates.

Use concise entries that capture what changed and why it matters. Reword entries during
review so this section stays useful for future troubleshooting.

This branch converts the MATLAB **ctFIRE** pipeline to Python + C++ (it does not develop
the napari plugin or the CurveAlign / curvelet ports — those already live in `main`).

- 2026-03-10 67ebfa5 feat: port CTrec_1.m → ct_reconstruction.py (curvelet preprocessing for FIRE)
- 2026-03-10 8fc74e4 feat: begin fire_2d_angle — FIRE 2D fiber-extraction port (through smoothing)
- 2026-04-07 74c35cd feat: add a corrected check_danglers that fixes a dead-code bug in MATLAB's check_danglers.m (kept gated off by default for MATLAB-faithful parity)
- 2026-04-26 12cdbd5 feat: port core FIRE routines to the C++ fiber_backend (local maxima, distance transform, fiberlinkgap, link extension)
- 2026-04-28 47ed821 refactor: split fiber analysis and angle modules
- 2026-05-13 f1191cf fix: settle on 0-based vertex indexing across trimxfv/analysis/processing (MATLAB parity)
- 2026-05-15 4adc2e3 fix: calc_fiberlen off-by-one and angle_xy axis swap, with MATLAB regression tests
- 2026-05-17 835a902 fix: correct row/col axis convention in calc_fiberang2
- 2026-05-28 659968d feat: assemble the ct_fire pipeline (ct_reconstruction + fire_2d) — 56 parity tests passing
- 2026-06-07 0c5e701 refactor: lazy-load ct_reconstruction inside functions (avoid hard curvelops import)
- 2026-06-07 db956ff fix: gracefully skip curvelops-dependent ct_fire tests/imports on CI
- 2026-06-19 fa6deb3 fix: correct row/col indexing and use-after-free in the FIRE 2D C++ backend
- 2026-07-06 aee3609 test: add non-square image handling tests for ct_fire and fire_2d_angle
- 2026-07-06 cc912a4 build: compile fiber_backend as a pybind11 extension during install
- 2026-07-06 b8b4dc9 fix: use matplotlib.colormaps API instead of deprecated get_cmap
- 2026-07-13 510e587 build: make the native C++ extension build optional; decouple GUI deps
