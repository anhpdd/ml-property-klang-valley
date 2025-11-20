"""
Tests for metrics utilities.
"""

import pytest
import numpy as np
from src.utils.metrics import (
    calculate_metrics,
    mean_absolute_percentage_error,
    calculate_price_accuracy
)


def test_calculate_metrics():
    """Test metrics calculation."""
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([110, 190, 310, 390, 510])

    metrics = calculate_metrics(y_true, y_pred, prefix="test", include_mape=True)

    assert 'test_mse' in metrics
    assert 'test_rmse' in metrics
    assert 'test_mae' in metrics
    assert 'test_r2' in metrics
    assert 'test_mape' in metrics

    # Check that all metrics are positive
    assert metrics['test_mse'] > 0
    assert metrics['test_rmse'] > 0
    assert metrics['test_mae'] > 0


def test_mean_absolute_percentage_error():
    """Test MAPE calculation."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 300])  # 10%, 5%, 0% errors

    mape = mean_absolute_percentage_error(y_true, y_pred)

    # MAPE should be around 5% (average of 10%, 5%, 0%)
    assert 4 < mape < 6


def test_calculate_price_accuracy():
    """Test price accuracy within tolerance."""
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([105, 210, 290, 420, 480])  # Errors: 5%, 5%, 3%, 5%, 4%

    accuracy = calculate_price_accuracy(y_true, y_pred, tolerance=0.10)

    assert 'accuracy_within_tolerance' in accuracy
    assert accuracy['accuracy_within_tolerance'] == 100.0  # All within 10%
