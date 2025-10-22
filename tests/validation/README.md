# Validation Tests

This directory contains tests that validate specific numerical outputs and MATLAB compatibility. These tests are **not included in CI** because they are:

## Why These Tests Are Not in CI

### **Brittle Numerical Comparisons**
- Test exact MATLAB reference values (e.g., `89.6518396°`, `5.625°`)
- Fail when algorithm improvements change outputs slightly
- Require specific tolerance values that may be too strict

### **Environment Dependencies**
- Require specific CurveLab builds and configurations
- Need manual environment setup
- Depend on external MATLAB reference data

### **Development-Focused**
- Validate MATLAB compatibility during development
- Test reconstruction quality with specific thresholds
- Verify exact algorithm behavior against reference implementations

## Test Files

### `test_relative_angles.py`
- Tests boundary analysis against MATLAB reference values
- 6 hardcoded test cases with specific angle expectations
- Validates `angle2boundaryEdge` calculations

### `test_new_curv.py`  
- Tests curvelet extraction against MATLAB parameters
- Validates exact angle increments (`5.625°`, `11.25°`)
- Tests MATLAB wedge count compatibility

### `test_curvelops_final.py`
- Tests CurveLab integration with quality metrics
- Validates reconstruction quality with MSE thresholds
- Tests synthetic fiber analysis with specific patterns

## Running Validation Tests

```bash
# Run all validation tests
pytest tests/validation/ -v

# Run specific validation test
pytest tests/validation/test_relative_angles.py -v

# Run with CurveLab environment
source setup_curvelops_env.sh
pytest tests/validation/test_curvelops_final.py -v
```

## When to Use These Tests

- **During development**: Validate MATLAB compatibility
- **Before releases**: Ensure numerical accuracy
- **Algorithm changes**: Verify specific behaviors
- **Debugging**: Compare against reference implementations

## CI vs Validation Test Strategy

- **CI Tests**: Focus on functionality, API structure, and integration
- **Validation Tests**: Focus on numerical accuracy and MATLAB compatibility
- **Separation**: Keeps CI fast and reliable while maintaining validation tools
