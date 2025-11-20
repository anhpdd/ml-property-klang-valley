# Tests

Test suite for the ML Property Valuation package.

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
```

## Test Structure

```
tests/
├── __init__.py
├── README.md (this file)
├── test_config.py          # Configuration tests
├── test_data_loaders.py    # Data loading tests
└── test_utils_metrics.py   # Metrics utilities tests
```

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

## Continuous Integration

Tests should be run automatically on all pull requests.

## TODO

- Add tests for geocoding module
- Add tests for clustering algorithms
- Add tests for model training pipeline
- Add integration tests for full pipeline
- Add tests for CLI scripts
