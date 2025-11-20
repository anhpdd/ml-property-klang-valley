"""
Tests for data loading utilities.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.data.loaders import get_data_summary


def test_get_data_summary():
    """Test data summary generation."""
    # Create sample DataFrame
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': ['a', 'b', 'c', 'd', 'e'],
        'col3': [1.1, 2.2, None, 4.4, 5.5]
    })

    summary = get_data_summary(df)

    assert isinstance(summary, dict)
    assert summary['n_rows'] == 5
    assert summary['n_columns'] == 3
    assert 'columns' in summary
    assert 'missing_values' in summary
    assert summary['missing_values']['col3'] == 1


# Add more tests as needed for other data functions
