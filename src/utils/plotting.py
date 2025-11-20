"""
Plotting utilities for visualizations.

Provides consistent styling for all project visualizations.
"""

import logging
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..config import COLORS, FIGURE_SIZE, DPI, PLOT_STYLE

logger = logging.getLogger(__name__)

# Set seaborn style
sns.set_theme(style=PLOT_STYLE)
plt.rcParams['figure.figsize'] = FIGURE_SIZE


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual",
    xlabel: str = "Sample Index (Sorted by Actual Value)",
    ylabel: str = "Price (RM/m²)",
    save_path: Optional[str] = None
) -> None:
    """
    Plot predictions vs actual values.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Optional path to save figure
    """
    logger.info(f"Plotting predictions for {len(y_true)} samples")

    # Create DataFrame and sort by actual
    plot_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    }).sort_values(by='Actual')

    # Create plot
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Plot predicted as line
    sns.lineplot(
        x=np.arange(len(plot_df)),
        y='Predicted',
        data=plot_df,
        color=COLORS['red'],
        label='Predicted Values',
        ax=ax,
        zorder=1
    )

    # Plot actual as scatter
    sns.scatterplot(
        x=np.arange(len(plot_df)),
        y='Actual',
        data=plot_df,
        label='Actual Values',
        ax=ax,
        zorder=2,
        alpha=0.6
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved plot to {save_path}")

    plt.show()


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 10,
    title: str = "Feature Importance",
    save_path: Optional[str] = None
) -> None:
    """
    Plot feature importance as horizontal bar chart.

    Args:
        feature_names: List of feature names
        importances: Feature importance values
        top_n: Number of top features to show
        title: Plot title
        save_path: Optional path to save figure
    """
    logger.info(f"Plotting feature importance (top {top_n})")

    # Create DataFrame and get top N
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Use gradient colors
    colors = plt.cm.Blues_r(np.linspace(0.4, 0.8, len(importance_df)))

    bars = ax.barh(
        range(len(importance_df)),
        importance_df['Importance'],
        color=colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )

    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['Feature'], fontsize=11)
    ax.set_xlabel('Feature Importance (Gini Impurity Reduction)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, importance_df['Importance'])):
        ax.text(
            value + 0.005,
            bar.get_y() + bar.get_height()/2,
            f'{value:.4f}',
            va='center',
            fontsize=10,
            fontweight='bold'
        )

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved feature importance plot to {save_path}")

    plt.show()


def plot_cluster_map(
    df: pd.DataFrame,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    cluster_col: str = 'market_cluster_id',
    price_col: str = 'price_m2',
    title: str = "Market Clusters by Price",
    save_path: Optional[str] = None
) -> None:
    """
    Plot property clusters on a map colored by price.

    Args:
        df: DataFrame with property data
        lat_col: Latitude column name
        lon_col: Longitude column name
        cluster_col: Cluster ID column name
        price_col: Price column for coloring
        title: Plot title
        save_path: Optional path to save figure

    Note:
        For interactive maps, use folium. This creates a static matplotlib plot.
        See notebooks/4_Clustering.ipynb for folium implementation.
    """
    logger.info(f"Plotting cluster map for {df[cluster_col].nunique()} clusters")

    fig, ax = plt.subplots(figsize=(14, 10))

    scatter = ax.scatter(
        df[lon_col],
        df[lat_col],
        c=df[price_col],
        cmap='RdYlGn_r',
        alpha=0.6,
        s=5
    )

    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Price (RM/m²)', fontsize=12)

    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved cluster map to {save_path}")

    plt.show()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot residual diagnostics.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        save_path: Optional path to save figure
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color=COLORS['red'], linestyle='--')
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted')
    axes[0].grid(True, linestyle='--', alpha=0.3)

    # Residual histogram
    axes[1].hist(residuals, bins=50, color=COLORS['blue'], alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color=COLORS['red'], linestyle='--')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    axes[1].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved residual plot to {save_path}")

    plt.show()


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = 'Test_Orig R2',
    save_path: Optional[str] = None
) -> None:
    """
    Plot model comparison bar chart.

    Args:
        results_df: DataFrame with model comparison results
        metric: Metric to plot
        save_path: Optional path to save figure
    """
    logger.info(f"Plotting model comparison by {metric}")

    # Sort by metric
    results_sorted = results_df.sort_values(metric, ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.barh(
        range(len(results_sorted)),
        results_sorted[metric],
        color=COLORS['blue'],
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )

    ax.set_yticks(range(len(results_sorted)))
    ax.set_yticklabels(results_sorted['Model'], fontsize=11)
    ax.set_xlabel(metric, fontsize=12)
    ax.set_title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for i, value in enumerate(results_sorted[metric]):
        ax.text(value + 0.01, i, f'{value:.4f}', va='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved model comparison plot to {save_path}")

    plt.show()
