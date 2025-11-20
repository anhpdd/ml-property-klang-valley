"""
Model training utilities.

Handles training of various regression models for property price prediction.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_validate
import xgboost as xgb
import lightgbm as lgb

from ..config import (
    RANDOM_STATE,
    CV_FOLDS,
    AVAILABLE_MODELS
)

logger = logging.getLogger(__name__)


def get_regression_models() -> Dict[str, any]:
    """
    Initialize and return all regression models for comparison.

    Returns:
        dict: Dictionary mapping model names to model instances
    """
    logger.info("Initializing regression models")

    models = {
        # Linear Models
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),

        # Non-Linear Models
        "KNeighbors Regressor": KNeighborsRegressor(),

        # Tree-Based Ensemble Models
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),

        # Advanced Gradient Boosting Models
        "XGBoost": xgb.XGBRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        ),
        "LightGBM": lgb.LGBMRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1
        )
    }

    return models


def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "Model"
) -> any:
    """
    Train a single model.

    Args:
        model: Scikit-learn compatible model instance
        X_train: Training features
        y_train: Training target (log-transformed)
        model_name: Name of model for logging

    Returns:
        Trained model instance
    """
    logger.info(f"Training {model_name}...")

    try:
        model.fit(X_train, y_train)
        logger.info(f"✅ {model_name} training complete")
        return model

    except Exception as e:
        logger.error(f"Failed to train {model_name}: {e}")
        raise


def cross_validate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = CV_FOLDS,
    model_name: str = "Model"
) -> Dict[str, any]:
    """
    Perform cross-validation on a model.

    Args:
        model: Scikit-learn compatible model instance
        X_train: Training features
        y_train: Training target (log-transformed)
        cv: Number of cross-validation folds
        model_name: Name of model for logging

    Returns:
        dict: Cross-validation results
    """
    logger.info(f"Cross-validating {model_name} with {cv}-fold CV...")

    kfold = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']

    cv_results = cross_validate(
        model, X_train, y_train,
        cv=kfold,
        scoring=scoring,
        n_jobs=-1
    )

    # Aggregate results
    results = {
        "cv_r2_mean": np.mean(cv_results['test_r2']),
        "cv_r2_std": np.std(cv_results['test_r2']),
        "cv_rmse_mean": np.sqrt(np.mean(-cv_results['test_neg_mean_squared_error'])),
        "cv_mae_mean": np.mean(-cv_results['test_neg_mean_absolute_error'])
    }

    logger.info(
        f"{model_name} CV results: "
        f"R²={results['cv_r2_mean']:.4f}±{results['cv_r2_std']:.4f}, "
        f"RMSE={results['cv_rmse_mean']:.4f}"
    )

    return results


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = CV_FOLDS,
    models_to_train: Optional[list] = None
) -> Tuple[Dict[str, any], Dict[str, dict]]:
    """
    Train and cross-validate all models.

    Args:
        X_train: Training features
        y_train: Training target (log-transformed)
        cv: Number of cross-validation folds
        models_to_train: List of model names to train (None = all)

    Returns:
        Tuple[Dict, Dict]: (trained_models, cv_results)
    """
    logger.info("Starting training for all models")

    all_models = get_regression_models()

    if models_to_train:
        all_models = {k: v for k, v in all_models.items() if k in models_to_train}

    trained_models = {}
    cv_results_all = {}

    for name, model in all_models.items():
        try:
            # Cross-validation
            cv_res = cross_validate_model(model, X_train, y_train, cv, name)
            cv_results_all[name] = cv_res

            # Final training on full training set
            trained_model = train_model(model, X_train, y_train, name)
            trained_models[name] = trained_model

        except Exception as e:
            logger.error(f"Failed to process {name}: {e}")
            continue

    logger.info(f"✅ Training complete for {len(trained_models)} models")

    return trained_models, cv_results_all


def save_model(
    model,
    output_path: str,
    model_name: str = "Model"
) -> None:
    """
    Save a trained model to disk.

    Args:
        model: Trained model instance
        output_path: Path to save the model
        model_name: Name of model for logging
    """
    import pickle

    logger.info(f"Saving {model_name} to {output_path}")

    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    logger.info(f"✅ {model_name} saved successfully")


def load_model(model_path: str) -> any:
    """
    Load a trained model from disk.

    Args:
        model_path: Path to saved model

    Returns:
        Loaded model instance
    """
    import pickle

    logger.info(f"Loading model from {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    logger.info("✅ Model loaded successfully")

    return model
