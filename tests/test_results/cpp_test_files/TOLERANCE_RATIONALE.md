# Tolerance Rationale for C++ Function Validation

This document defines and justifies the numerical tolerances used when comparing C++ outputs against MATLAB reference data.

## Overview

Tolerances are necessary because:

1. **Float32 precision limits**: Both MATLAB and C++ use float32 for distance maps, leading to minor rounding differences
2. **Different random number generators**: MATLAB uses Mersenne Twister; C++ uses a custom fastrand implementation
3. **Parallelization**: OpenMP parallel loops may process data in different orders
4. **Coordinate system conversions**: 1-based (MATLAB) ↔ 0-based (C++) conversions can introduce off-by-one errors

All comparisons use `np.testing.assert_allclose(pred, ref, rtol=X, atol=Y)` where:

- `rtol`: Relative tolerance (percentage difference)
- `atol`: Absolute tolerance (fixed difference)
- Both conditions must be satisfied: `|pred - ref| ≤ atol + rtol * |ref|`

## Tolerance Definitions

### 1. Integer Coordinates (Nucleation Points, Vertex Indices)

**Tolerance**: `rtol=0.0, atol=1.0`

**Rationale**:

- Coordinates should match within 1 pixel
- The random perturbation in `findlocmax` (epsilon * rand) can shift local maxima detection by ±1 pixel
- Parallelization may cause different tie-breaking order, but the same local maxima should be found

**Expected Differences**:

- ✓ Nucleation points shifted by ±1 pixel due to random tie-breaking
- ✓ Different ordering of nucleation points (OpenMP parallel gather)
- ✗ Missing/extra nucleation points (indicates a bug)
- ✗ Shifts > 1 pixel (indicates indexing bug)

**Test Method**:

- Sort both MATLAB and C++ outputs by coordinates (lexsort)
- Compare sorted arrays element-wise
- Allow ±1 pixel difference in each dimension

**Example**:

```python
# MATLAB: [[100, 200, 1], [150, 250, 1]]
# C++:    [[100, 201, 1], [150, 250, 1]]
# Difference: [0, 1, 0] - ACCEPTABLE (within atol=1.0)
```

### 2. Float Coordinates (Distance Map Values, Radii)

**Tolerance**: `rtol=1e-5, atol=1e-6`

**Rationale**:

- Float32 has ~7 decimal digits of precision
- Distance transform and smoothing operations accumulate rounding errors
- Multiple arithmetic operations can compound precision loss
- `rtol=1e-5` allows 0.001% relative error (well within float32 precision)
- `atol=1e-6` catches absolute differences for very small values near zero

**Expected Differences**:

- ✓ Minor precision differences (< 0.001%) from float arithmetic
- ✓ Slightly different smoothing results due to convolution order
- ✗ Large differences (> 0.01%) indicate algorithmic discrepancy

**Test Method**:

```python
np.testing.assert_allclose(cpp_dsm, matlab_dsm, rtol=1e-5, atol=1e-6)
```

**Example**:

```python
# MATLAB: 3.456789
# C++:    3.456792
# Difference: 0.000003 (~0.0001%) - ACCEPTABLE
```

### 3. Fiber Counts

**Tolerance**: `rtol=0.1, atol=5`

**Rationale**:

- Fiber counts can differ due to:
  1. Different tie-breaking in LMP detection (±1-2 fibers per nucleation point)
  2. Parallel processing order affecting duplicate fiber removal
  3. Minor differences in extension threshold application
- `rtol=0.1` allows 10% variation in fiber count
- `atol=5` ensures at least 5 fibers difference is tolerated for small networks

**Expected Differences**:

- ✓ 5-15% difference in fiber count (due to tie-breaking, duplicate removal)
- ✗ > 20% difference indicates missing extension logic or bug
- ✗ Exact same count expected if random seed is properly matched

**Test Method**:

```python
assert abs(len(cpp_F) - len(matlab_F)) <= max(5, 0.1 * len(matlab_F))
```

**Example**:

```python
# MATLAB: 1672 fibers
# C++:    1620 fibers (52 fewer = 3.1%)
# Acceptable if within 10% (167 fibers)
```

### 4. Graph Connectivity (F, V structures)

**Tolerance**: **Exact match after sorting**

**Rationale**:

- Graph structure (which vertices connect to which fibers) must be identical
- Ordering may differ due to parallelization
- Vertex indices must match (accounting for potential renumbering)

**Expected Differences**:

- ✓ Different ordering of fibers in `F` array
- ✓ Different ordering of vertex indices within each fiber
- ✗ Different fiber connectivity (vertices connected in MATLAB but not C++)
- ✗ Missing vertices or fibers

**Test Method**:

1. Sort fibers by their first vertex
2. Sort vertices within each fiber
3. Compare sorted structures for exact match

**Example**:

```python
# MATLAB F[0].v = [1, 5, 10, 15]
# C++    F[0].v = [1, 5, 10, 15]  ✓ EXACT MATCH
# C++    F[0].v = [1, 5, 11, 15]  ✗ DIFFERENT CONNECTIVITY (Bug!)
```

### 5. Radius Values (R array)

**Tolerance**: `rtol=0.01, atol=0.1`

**Rationale**:

- Radii are computed from distance map values: `R = ceil(d(x, y))`
- Distance map values may differ slightly (±0.001)
- `ceil()` operation is sensitive to values near integers
- `rtol=0.01` allows 1% relative error
- `atol=0.1` allows 0.1 pixel absolute error (less than ceil rounding)

**Expected Differences**:

- ✓ ±0.1 pixel difference in radii
- ✗ Differences > 0.5 pixels (indicates distance map discrepancy)

**Test Method**:

```python
np.testing.assert_allclose(cpp_R, matlab_R, rtol=0.01, atol=0.1)
```

### 6. Vertex Connectivity Lists (V.f, V.fe, V.vall)

**Tolerance**: **Set equality (after sorting)**

**Rationale**:

- `V[i].f`: List of fiber indices passing through vertex i
- `V[i].fe`: List of fiber indices with endpoint at vertex i
- `V[i].vall`: List of all vertices in connected fibers
- Lists may be in different order but must contain same elements

**Expected Differences**:

- ✓ Different ordering of indices in lists
- ✗ Missing or extra indices (connectivity bug)

**Test Method**:

```python
def compare_vertex_lists(cpp_v, matlab_v):
    assert set(cpp_v['f']) == set(matlab_v['f'])
    assert set(cpp_v['fe']) == set(matlab_v['fe'])
    assert set(cpp_v['vall']) == set(matlab_v['vall'])
```

## Special Cases

### Random Seed Reproducibility

For exact reproducibility tests where random seed should match MATLAB:

**Tolerance**: `rtol=0.0, atol=0` (exact match expected)

**Conditions**:

- Both MATLAB and C++ set seed to 100
- Same random number generator algorithm used
- No parallelization (or deterministic parallel execution)

**If this fails**: Document as "expected difference due to RNG implementation"

### Padding/Coordinate Transform

**Tolerance**: `rtol=0.0, atol=0` (exact match expected)

**Rationale**:

- MATLAB `extend_xlink` pads volume by zpad=3
- All coordinates should be shifted consistently
- No tolerance needed for integer shifts

**Test Method**:

```python
# Check that all coordinates are shifted by zpad=3
assert np.all(cpp_X == matlab_X)  # After accounting for padding
```

## Adjusting Tolerances

### When to Tighten Tolerances

- If tests are passing with large margins (e.g., differences are 0.01% but tolerance is 1%)
- After fixing bugs to ensure no regression
- For deterministic operations (coordinate transforms, integer arithmetic)

### When to Loosen Tolerances

- If tests fail due to known, acceptable differences (document reason)
- For operations with cumulative rounding errors (multiple smoothing passes)
- When parallelization introduces non-determinism

### Never Acceptable

These differences always indicate bugs:

- Missing vertices or fibers (not just reordering)
- Coordinate shifts > 1 pixel (indexing bugs)
- Connectivity changes (different graph structure)
- Large float differences (> 0.1% for most operations)

## Testing Guidelines

1. **Always test with sorted data** to eliminate ordering differences
2. **Document expected differences** in test docstrings
3. **Use tight tolerances by default**, loosen only with justification
4. **Run tests multiple times** to check for non-determinism
5. **Compare intermediate values** not just final outputs
6. **Profile performance** to ensure parallelization is beneficial

## References

- MATLAB `findlocmax`: Uses `rng(100, 'twister')` and `epsilon * rand(size(d))`
- C++ `findlocmax_native`: Uses `fastrand` with integer seed
- MATLAB `extend_xlink`: Pads by zpad=3, uses float64 internally
- C++ `extend_xlink_native`: Uses float32, parallelizes over nucleation points

## Revision History

- **2026-04-11**: Initial tolerance definitions for validation testing
- Future: Update based on validation results and bug fixes

