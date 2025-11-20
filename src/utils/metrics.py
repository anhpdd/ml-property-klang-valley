"""
Custom metrics and evaluation utilities.

Provides specialized metrics for property price prediction.
"""

import logging
from typing import Dict

import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error as sklearn_mape
)

logger = logging.getLogger(__name__)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        float: MAPE as a percentage (0-100)

    Note:
        MAPE should ONLY be calculated in original scale (RM/m²), NOT in log-space.
    """
    mape = sklearn_mape(y_true, y_pred) * 100
    return mape


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
    include_mape: bool = True
) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        prefix: Prefix for metric names (e.g., "Train", "Test")
        include_mape: Whether to include MAPE (only for original scale)

    Returns:
        dict: Dictionary of all metrics
    """
    prefix = f"{prefix}_" if prefix else ""

    metrics = {
        f"{prefix}mse": mean_squared_error(y_true, y_pred),
        f"{prefix}rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        f"{prefix}mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}r2": r2_score(y_true, y_pred)
    }

    if include_mape:
        metrics[f"{prefix}mape"] = mean_absolute_percentage_error(y_true, y_pred)

    return metrics


def calculate_price_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: float = 0.15
) -> Dict[str, float]:
    """
    Calculate percentage of predictions within tolerance.

    Args:
        y_true: Actual prices
        y_pred: Predicted prices
        tolerance: Acceptable error tolerance (default 15%)

    Returns:
        dict: Accuracy metrics
    """
    errors = np.abs(y_true - y_pred) / y_true

    within_tolerance = (errors <= tolerance).sum()
    accuracy_pct = within_tolerance / len(y_true) * 100

    return {
        'accuracy_within_tolerance': accuracy_pct,
        'tolerance': tolerance * 100,
        'n_within_tolerance': within_tolerance,
        'n_total': len(y_true)
    }


def calculate_price_bands_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bands: list = None
) -> Dict[str, dict]:
    """
    Calculate accuracy metrics by price bands.

    Args:
        y_true: Actual prices
        y_pred: Predicted prices
        bands: List of (min, max) tuples for price bands

    Returns:
        dict: Metrics for each price band
    """
    if bands is None:
        bands = [
            (0, 3000, 'Budget'),
            (3000, 8000, 'Mid-tier'),
            (8000, np.inf, 'Luxury')
        ]

    results = {}

    for min_price, max_price, name in bands:
        mask = (y_true >= min_price) & (y_true < max_price)

        if mask.sum() == 0:
            continue

        y_true_band = y_true[mask]
        y_pred_band = y_pred[mask]

        metrics = calculate_metrics(y_true_band, y_pred_band, include_mape=True)
        metrics['n_properties'] = mask.sum()

        results[name] = metrics

    return results


def residual_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, any]:
    """
    Analyze prediction residuals.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        dict: Residual statistics
    """
    residuals = y_true - y_pred

    analysis = {
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'min_residual': np.min(residuals),
        'max_residual': np.max(residuals),
        'median_residual': np.median(residuals),
        'q25_residual': np.percentile(residuals, 25),
        'q75_residual': np.percentile(residuals, 75),
        'n_overestimated': (residuals < 0).sum(),  # pred > actual
        'n_underestimated': (residuals > 0).sum(),  # pred < actual
    }

    return analysis
