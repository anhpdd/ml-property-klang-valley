"""
Model training, evaluation, and prediction modules.

Handles ML model training, performance evaluation, and making predictions
on new property data.
"""

from .trainer import train_model, get_regression_models, train_all_models
from .evaluator import evaluate_model, calculate_metrics, compare_models
from .predictor import PropertyPredictor, predict_property_price

__all__ = [
    'train_model',
    'get_regression_models',
    'train_all_models',
    'evaluate_model',
    'calculate_metrics',
    'compare_models',
    'PropertyPredictor',
    'predict_property_price'
]
