# Validation Tests

Optional parity checks against historical MATLAB reference outputs live in
`pycurvelets-core/tests/validation/`. These checks are useful during calibration work, but they
are intentionally kept out of default CI because they depend on a specific
CurveLab-enabled environment and strict numerical matching to reference
artifacts. Small implementation changes can legitimately shift values by tiny
amounts while preserving behavior, so these tests are best run explicitly when
you are validating parity, not on every pull request.

Run validation tests locally when needed:

```bash
pytest pycurvelets-core/tests/validation/ -v
```

Run a specific validation module:

```bash
pytest pycurvelets-core/tests/validation/test_relative_angles.py -v
```

If the test requires CurveLab runtime setup:

```bash
source setup_curvelops_env.sh
pytest pycurvelets-core/tests/validation/test_curvelops_final.py -v
```
