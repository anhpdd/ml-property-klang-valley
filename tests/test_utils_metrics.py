"""
Tests for metrics utilities.
"""

import pytest
import numpy as np
from src.utils.metrics import (
    calculate_metrics,
    mean_absolute_percentage_error,
    calculate_price_accuracy,
    residual_analysis
)


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_calculate_metrics_basic(self):
        """Test basic metrics calculation."""
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

    def test_calculate_metrics_without_mape(self):
        """Test metrics calculation without MAPE."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])

        metrics = calculate_metrics(y_true, y_pred, include_mape=False)

        assert 'mape' not in metrics
        assert 'r2' in metrics

    def test_calculate_metrics_perfect_prediction(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])

        metrics = calculate_metrics(y_true, y_pred, include_mape=True)

        assert metrics['mse'] == 0
        assert metrics['rmse'] == 0
        assert metrics['mae'] == 0
        assert metrics['r2'] == 1.0
        assert metrics['mape'] == 0


class TestMAPE:
    """Tests for MAPE calculation with zero handling."""

    def test_mape_basic(self):
        """Test basic MAPE calculation."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 300])  # 10%, 5%, 0% errors

        mape = mean_absolute_percentage_error(y_true, y_pred)

        # MAPE should be around 5% (average of 10%, 5%, 0%)
        assert 4 < mape < 6

    def test_mape_with_zeros_exclude(self):
        """Test MAPE excludes zero values by default."""
        y_true = np.array([100, 0, 200])  # Contains zero
        y_pred = np.array([110, 10, 190])

        # Should not raise, zeros excluded
        mape = mean_absolute_percentage_error(y_true, y_pred, zero_handling='exclude')

        # Only non-zero values used: (10% + 5%) / 2 = 7.5%
        assert 6 < mape < 9

    def test_mape_with_zeros_raise(self):
        """Test MAPE raises error when zeros present and mode is 'raise'."""
        y_true = np.array([100, 0, 200])
        y_pred = np.array([110, 10, 190])

        with pytest.raises(ValueError, match="zero/near-zero values"):
            mean_absolute_percentage_error(y_true, y_pred, zero_handling='raise')

    def test_mape_with_zeros_epsilon(self):
        """Test MAPE with epsilon replacement for zeros."""
        y_true = np.array([100, 0, 200])
        y_pred = np.array([110, 10, 190])

        # Should not raise
        mape = mean_absolute_percentage_error(y_true, y_pred, zero_handling='epsilon')

        assert mape > 0

    def test_mape_all_zeros_raises(self):
        """Test MAPE raises when all values are zero."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([10, 20, 30])

        with pytest.raises(ValueError, match="All values are zero"):
            mean_absolute_percentage_error(y_true, y_pred, zero_handling='exclude')

    def test_mape_invalid_mode_raises(self):
        """Test MAPE raises for invalid zero handling mode."""
        # Must include zero to trigger the handling code path
        y_true = np.array([100, 0, 200])
        y_pred = np.array([110, 10, 190])

        with pytest.raises(ValueError, match="Unknown zero_handling mode"):
            mean_absolute_percentage_error(y_true, y_pred, zero_handling='invalid')


class TestPriceAccuracy:
    """Tests for price accuracy calculation."""

    def test_price_accuracy_basic(self):
        """Test basic price accuracy within tolerance."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([105, 210, 290, 420, 480])  # All within 10%

        accuracy = calculate_price_accuracy(y_true, y_pred, tolerance=0.10)

        assert 'accuracy_within_tolerance' in accuracy
        assert accuracy['accuracy_within_tolerance'] == 100.0

    def test_price_accuracy_partial(self):
        """Test price accuracy with some outside tolerance."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 400])  # Last one is 33% off

        accuracy = calculate_price_accuracy(y_true, y_pred, tolerance=0.15)

        # 2 out of 3 within 15%
        assert accuracy['accuracy_within_tolerance'] == pytest.approx(66.67, rel=0.01)

    def test_price_accuracy_with_zeros(self):
        """Test price accuracy excludes zero values."""
        y_true = np.array([100, 0, 200])  # Contains zero
        y_pred = np.array([110, 50, 210])

        # Should not raise, zeros excluded
        accuracy = calculate_price_accuracy(y_true, y_pred, tolerance=0.15)

        assert accuracy['n_total'] == 2  # Zero excluded

    def test_price_accuracy_all_zeros_raises(self):
        """Test price accuracy raises when all values are zero."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([10, 20, 30])

        with pytest.raises(ValueError, match="All y_true values are zero"):
            calculate_price_accuracy(y_true, y_pred)


class TestResidualAnalysis:
    """Tests for residual analysis."""

    def test_residual_analysis_basic(self):
        """Test basic residual analysis."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 300])  # -10, +10, 0

        analysis = residual_analysis(y_true, y_pred)

        assert 'mean_residual' in analysis
        assert 'std_residual' in analysis
        assert 'n_overestimated' in analysis
        assert 'n_underestimated' in analysis

        assert analysis['mean_residual'] == 0  # (-10 + 10 + 0) / 3
        assert analysis['n_overestimated'] == 1  # pred > actual once
        assert analysis['n_underestimated'] == 1  # pred < actual once
