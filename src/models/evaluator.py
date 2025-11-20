"""
Model evaluation utilities.

Handles metrics calculation, model comparison, and performance visualization.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

from ..config import COLORS

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
    include_mape: bool = True
) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        prefix: Prefix for metric names (e.g., "Train_Log", "Test_Orig")
        include_mape: Whether to include MAPE (only for original scale, not log space)

    Returns:
        dict: Dictionary of metrics

    Note:
        MAPE should only be calculated in original scale (RM/m²), NOT in log-space.
    """
    prefix = f"{prefix} " if prefix else ""

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        f"{prefix}MSE": mse,
        f"{prefix}RMSE": rmse,
        f"{prefix}MAE": mae,
        f"{prefix}R2": r2
    }

    # Only calculate MAPE if requested (for original space, not log space)
    if include_mape:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # Convert to percentage
        metrics[f"{prefix}MAPE"] = mape

    return metrics


def evaluate_model(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_log: np.ndarray,
    y_test_log: np.ndarray,
    y_train_orig: np.ndarray,
    y_test_orig: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate a trained model on train and test sets.

    Args:
        model: Trained model instance
        X_train: Training features
        X_test: Test features
        y_train_log: Training target (log-space)
        y_test_log: Test target (log-space)
        y_train_orig: Training target (original scale)
        y_test_orig: Test target (original scale)
        model_name: Name of model for logging

    Returns:
        dict: All evaluation metrics
    """
    logger.info(f"Evaluating {model_name}...")

    # Make predictions in log-space
    y_train_pred_log = model.predict(X_train)
    y_test_pred_log = model.predict(X_test)

    # Inverse transform to original scale
    y_train_pred_orig = np.expm1(y_train_pred_log)
    y_test_pred_orig = np.expm1(y_test_pred_log)

    # Calculate metrics in log-space (NO MAPE)
    train_metrics_log = calculate_metrics(
        y_train_log, y_train_pred_log,
        prefix="Train_Log", include_mape=False
    )
    test_metrics_log = calculate_metrics(
        y_test_log, y_test_pred_log,
        prefix="Test_Log", include_mape=False
    )

    # Calculate metrics in original scale (WITH MAPE)
    train_metrics_orig = calculate_metrics(
        y_train_orig, y_train_pred_orig,
        prefix="Train_Orig", include_mape=True
    )
    test_metrics_orig = calculate_metrics(
        y_test_orig, y_test_pred_orig,
        prefix="Test_Orig", include_mape=True
    )

    # Combine all metrics
    all_metrics = {
        **train_metrics_log,
        **test_metrics_log,
        **train_metrics_orig,
        **test_metrics_orig
    }

    logger.info(
        f"{model_name} - Test R²: {all_metrics['Test_Orig R2']:.4f}, "
        f"MAPE: {all_metrics['Test_Orig MAPE']:.2f}%"
    )

    return all_metrics


def compare_models(
    trained_models: Dict[str, any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_log: np.ndarray,
    y_test_log: np.ndarray,
    y_train_orig: np.ndarray,
    y_test_orig: np.ndarray
) -> pd.DataFrame:
    """
    Compare performance of multiple models.

    Args:
        trained_models: Dictionary of trained models
        X_train: Training features
        X_test: Test features
        y_train_log: Training target (log-space)
        y_test_log: Test target (log-space)
        y_train_orig: Training target (original scale)
        y_test_orig: Test target (original scale)

    Returns:
        pd.DataFrame: Comparison table sorted by Test R²
    """
    logger.info(f"Comparing {len(trained_models)} models...")

    results = []

    for name, model in trained_models.items():
        metrics = evaluate_model(
            model, X_train, X_test,
            y_train_log, y_test_log,
            y_train_orig, y_test_orig,
            name
        )
        metrics['Model'] = name
        results.append(metrics)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by Test R² (original scale)
    df = df.sort_values('Test_Orig R2', ascending=False).reset_index(drop=True)
    df['Ranking'] = df.index + 1

    # Reorder columns
    cols = ['Ranking', 'Model',
            'Train_Log R2', 'Test_Log R2',
            'Train_Log RMSE', 'Test_Log RMSE',
            'Train_Log MAE', 'Test_Log MAE',
            'Train_Orig R2', 'Test_Orig R2',
            'Train_Orig RMSE', 'Test_Orig RMSE',
            'Train_Orig MAE', 'Test_Orig MAE',
            'Train_Orig MAPE', 'Test_Orig MAPE']

    df = df[[c for c in cols if c in df.columns]]

    logger.info(f"✅ Model comparison complete")
    logger.info(f"Top 3 models by Test R²:")
    for idx, row in df.head(3).iterrows():
        logger.info(f"  {idx+1}. {row['Model']}: R²={row['Test_Orig R2']:.4f}, MAPE={row['Test_Orig MAPE']:.2f}%")

    return df


def plot_model_performance(
    model_name: str,
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    y_train_original: np.ndarray,
    y_train_pred_original: np.ndarray,
    y_test_original: np.ndarray,
    y_test_pred_original: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot model performance in both log-space and original scale.

    Args:
        model_name: Name of the model
        y_train, y_train_pred: Training actuals and predictions in LOG SPACE
        y_test, y_test_pred: Test actuals and predictions in LOG SPACE
        y_train_original, y_train_pred_original: Training actuals and predictions in ORIGINAL SCALE
        y_test_original, y_test_pred_original: Test actuals and predictions in ORIGINAL SCALE
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Performance Analysis for: {model_name}', fontsize=16, y=0.995)

    # ===== ROW 1: LOG SPACE =====
    plot_df_train_log = pd.DataFrame({
        'Actual': y_train,
        'Predicted': y_train_pred
    }).sort_values(by='Actual')

    plot_df_test_log = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_test_pred
    }).sort_values(by='Actual')

    # Training Set - Log Space
    sns.lineplot(x=np.arange(len(plot_df_train_log)), y='Predicted',
                data=plot_df_train_log, color=COLORS['red'], label='Predicted Values',
                ax=axes[0, 0], zorder=1)
    sns.scatterplot(x=np.arange(len(plot_df_train_log)), y='Actual',
                   data=plot_df_train_log, label='Actual Values',
                   ax=axes[0, 0], zorder=2, alpha=0.6)
    axes[0, 0].set_title('Training Set - Log Space')
    axes[0, 0].set_xlabel('Sample Index (Sorted by Actual Value)')
    axes[0, 0].set_ylabel('Log(Price)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)

    # Test Set - Log Space
    sns.lineplot(x=np.arange(len(plot_df_test_log)), y='Predicted',
                data=plot_df_test_log, color=COLORS['red'], label='Predicted Values',
                ax=axes[0, 1], zorder=1)
    sns.scatterplot(x=np.arange(len(plot_df_test_log)), y='Actual',
                   data=plot_df_test_log, label='Actual Values',
                   ax=axes[0, 1], zorder=2, alpha=0.6)
    axes[0, 1].set_title('Test Set - Log Space')
    axes[0, 1].set_xlabel('Sample Index (Sorted by Actual Value)')
    axes[0, 1].set_ylabel('Log(Price)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)

    # ===== ROW 2: ORIGINAL SCALE (RM/sqm) =====
    plot_df_train_orig = pd.DataFrame({
        'Actual': y_train_original,
        'Predicted': y_train_pred_original
    }).sort_values(by='Actual')

    plot_df_test_orig = pd.DataFrame({
        'Actual': y_test_original,
        'Predicted': y_test_pred_original
    }).sort_values(by='Actual')

    # Training Set - Original Scale
    sns.lineplot(x=np.arange(len(plot_df_train_orig)), y='Predicted',
                data=plot_df_train_orig, color=COLORS['red'], label='Predicted Values',
                ax=axes[1, 0], zorder=1)
    sns.scatterplot(x=np.arange(len(plot_df_train_orig)), y='Actual',
                   data=plot_df_train_orig, label='Actual Values',
                   ax=axes[1, 0], zorder=2, alpha=0.6)
    axes[1, 0].set_title('Training Set - Original Scale (RM/sqm)')
    axes[1, 0].set_xlabel('Sample Index (Sorted by Actual Value)')
    axes[1, 0].set_ylabel('Price (RM/sqm)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)

    # Test Set - Original Scale
    sns.lineplot(x=np.arange(len(plot_df_test_orig)), y='Predicted',
                data=plot_df_test_orig, color=COLORS['red'], label='Predicted Values',
                ax=axes[1, 1], zorder=1)
    sns.scatterplot(x=np.arange(len(plot_df_test_orig)), y='Actual',
                   data=plot_df_test_orig, label='Actual Values',
                   ax=axes[1, 1], zorder=2, alpha=0.6)
    axes[1, 1].set_title('Test Set - Original Scale (RM/sqm)')
    axes[1, 1].set_xlabel('Sample Index (Sorted by Actual Value)')
    axes[1, 1].set_ylabel('Price (RM/sqm)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved plot to {save_path}")

    plt.show()


def plot_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 10,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Plot feature importance for tree-based models.

    Args:
        model: Trained tree-based model (RandomForest, XGBoost, etc.)
        feature_names: List of feature names
        top_n: Number of top features to plot
        save_path: Optional path to save figure

    Returns:
        pd.DataFrame: Feature importance table
    """
    logger.info(f"Plotting feature importance (top {top_n})...")

    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        logger.error("Model does not have feature_importances_ attribute")
        return None

    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # Get top N
    top_features = importance_df.head(top_n)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.Blues_r(np.linspace(0.4, 0.8, len(top_features)))
    bars = ax.barh(range(len(top_features)), top_features['Importance'],
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'], fontsize=11)
    ax.set_xlabel('Feature Importance (Gini Impurity Reduction)', fontsize=12)
    ax.set_title(
        f'What Drives Property Prices in Klang Valley?\nTop {top_n} Features',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.invert_yaxis()

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_features['Importance'])):
        ax.text(value + 0.005, bar.get_y() + bar.get_height()/2,
                f'{value:.4f}',
                va='center', fontsize=10, fontweight='bold')

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved feature importance plot to {save_path}")

    plt.show()

    return importance_df
