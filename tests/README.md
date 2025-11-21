# Tests

Test suite for the ML Property Valuation package.

**Current Status:** 50 tests passing, 7 skipped (ML dependencies)

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_config.py

# Run with verbose output
pytest tests/ -v

# Run only security tests
pytest tests/test_security.py -v
```

## Test Structure

```
tests/
├── __init__.py
├── README.md                 # This file
├── test_config.py            # Configuration tests (4 tests)
├── test_data_loaders.py      # Data loading tests (1 test)
├── test_utils_metrics.py     # Metrics & MAPE tests (14 tests)
├── test_security.py          # Security validation tests (15 tests)
├── test_preprocessing.py     # Data preprocessing tests (10 tests)
└── test_osm.py               # OSM API retry logic tests (13 tests)
```

## Test Categories

### Security Tests (`test_security.py`)
- **Path Traversal Prevention:** Validates file paths stay within project
- **File Size Limits:** Tests memory exhaustion protection
- **Feature Validation:** Tests NaN/Inf rejection, feature count validation
- **Hash Verification:** Tests model integrity checks
- **Input Validation:** Tests handling of invalid file types, missing files

### Preprocessing Tests (`test_preprocessing.py`)
- **Division by Zero:** Tests safe handling of `land_m2 = 0`
- **Negative Values:** Tests handling of negative land areas and unit levels
- **NaN Handling:** Tests proper propagation of missing values
- **Inverse Transform:** Tests log-space to original scale conversion

### Metrics Tests (`test_utils_metrics.py`)
- **MAPE Zero Handling:** Tests exclude/epsilon/raise modes for zero values
- **Price Accuracy:** Tests tolerance-based accuracy calculation
- **Residual Analysis:** Tests prediction error analysis

### OSM Tests (`test_osm.py`)
- **Transient Error Detection:** Tests identification of retryable errors
- **Retry Logic:** Tests exponential backoff and max retries
- **Exception Handling:** Tests that KeyboardInterrupt/SystemExit pass through

## Adding New Tests

1. Create a new test file starting with `test_`
2. Import the module you want to test
3. Write test functions starting with `test_`
4. Use pytest fixtures for setup/teardown if needed

Example:

```python
import pytest
from src.module import function_to_test

def test_function_behavior():
    """Test that function works correctly."""
    result = function_to_test(input_data)
    assert result == expected_output
```

## Test Coverage

Run coverage report to identify untested code:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Skipped Tests

7 tests are skipped when ML dependencies (lightgbm, xgboost) are not installed:
- `TestFeatureValidation` (4 tests) - requires predictor module
- `TestHashVerification` (3 tests) - requires predictor module

To run all tests, install ML dependencies:
```bash
pip install lightgbm xgboost scikit-learn
```

## Continuous Integration

Tests should be run automatically on all pull requests.

## TODO

- [ ] Add tests for geocoding module
- [ ] Add tests for clustering algorithms
- [ ] Add tests for model training pipeline
- [ ] Add integration tests for full pipeline
- [ ] Add tests for CLI scripts
- [x] ~~Add security tests~~ ✅ Done
- [x] ~~Add preprocessing tests~~ ✅ Done
- [x] ~~Add OSM retry logic tests~~ ✅ Done
- [x] ~~Add MAPE zero handling tests~~ ✅ Done
