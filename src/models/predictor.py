"""
Property price prediction interface.

Provides a high-level API for making predictions on new property data.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import pickle

from ..config import PRODUCTION_MODEL, SCALER_MODEL

logger = logging.getLogger(__name__)


class PropertyPredictor:
    """
    High-level interface for property price prediction.

    Handles model loading, preprocessing, and prediction in a single interface.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize PropertyPredictor.

        Args:
            model_path: Path to trained model pickle file (defaults to config.PRODUCTION_MODEL)
            scaler_path: Path to fitted scaler pickle file (defaults to config.SCALER_MODEL)
        """
        self.model_path = Path(model_path) if model_path else PRODUCTION_MODEL
        self.scaler_path = Path(scaler_path) if scaler_path else SCALER_MODEL

        self.model = None
        self.scaler = None

        logger.info("PropertyPredictor initialized")

    def load_model(self) -> None:
        """Load trained model and scaler from disk."""
        logger.info(f"Loading model from {self.model_path}")

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                f"Train a model first using notebooks/5_Modelling.ipynb"
            )

        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

        logger.info(f"Loading scaler from {self.scaler_path}")

        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler file not found: {self.scaler_path}. "
                f"Train a model first using notebooks/5_Modelling.ipynb"
            )

        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        logger.info("✅ Model and scaler loaded successfully")

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_log_space: bool = False
    ) -> np.ndarray:
        """
        Predict property prices.

        Args:
            X: Features (must be preprocessed with exactly 279 features)
            return_log_space: If True, return predictions in log-space

        Returns:
            np.ndarray: Predicted prices (RM/m²)
        """
        if self.model is None or self.scaler is None:
            logger.info("Model not loaded, loading now...")
            self.load_model()

        logger.info(f"Making predictions for {len(X)} properties...")

        # Make predictions in log-space
        y_pred_log = self.model.predict(X)

        if return_log_space:
            logger.info("Returning predictions in log-space")
            return y_pred_log

        # Inverse transform to original scale (RM/m²)
        y_pred = np.expm1(y_pred_log)

        logger.info(f"✅ Predictions complete: mean price = RM {y_pred.mean():,.2f}/m²")

        return y_pred

    def predict_single(
        self,
        property_features: Dict[str, any]
    ) -> float:
        """
        Predict price for a single property.

        Args:
            property_features: Dictionary of property features

        Returns:
            float: Predicted price in RM/m²

        Example:
            >>> predictor = PropertyPredictor()
            >>> features = {
            ...     'property_m2': 1000,
            ...     'unit_level': 25,
            ...     'dist_to_rail_station': 0.3,
            ...     # ... all 279 features
            ... }
            >>> price = predictor.predict_single(features)
        """
        # Convert to DataFrame
        df = pd.DataFrame([property_features])

        # Predict
        prediction = self.predict(df)

        return prediction[0]

    def batch_predict(
        self,
        properties_df: pd.DataFrame,
        price_col: str = 'predicted_price_m2'
    ) -> pd.DataFrame:
        """
        Predict prices for multiple properties and add to DataFrame.

        Args:
            properties_df: DataFrame with property features
            price_col: Column name for predictions

        Returns:
            pd.DataFrame: DataFrame with predictions added
        """
        logger.info(f"Batch predicting for {len(properties_df)} properties...")

        # Make predictions
        predictions = self.predict(properties_df)

        # Add to DataFrame
        properties_df = properties_df.copy()
        properties_df[price_col] = predictions

        logger.info(f"✅ Batch prediction complete")

        return properties_df


def predict_property_price(
    X: Union[pd.DataFrame, np.ndarray],
    model_path: Optional[Union[str, Path]] = None,
    scaler_path: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """
    Convenience function to predict property prices.

    Args:
        X: Features (preprocessed)
        model_path: Path to trained model (optional)
        scaler_path: Path to fitted scaler (optional)

    Returns:
        np.ndarray: Predicted prices in RM/m²

    Example:
        >>> from src.models import predict_property_price
        >>> predictions = predict_property_price(X_test)
    """
    predictor = PropertyPredictor(model_path, scaler_path)
    return predictor.predict(X)


def load_and_predict(
    data_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    scaler_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Load data, make predictions, and optionally save results.

    Args:
        data_path: Path to CSV/Excel file with property data
        model_path: Path to trained model (optional)
        scaler_path: Path to fitted scaler (optional)
        output_path: Path to save predictions (optional)

    Returns:
        pd.DataFrame: DataFrame with predictions

    Example:
        >>> results = load_and_predict('data/new_properties.csv', output_path='predictions.csv')
    """
    from ..data import load_raw_data

    logger.info(f"Loading data from {data_path}")
    df = load_raw_data(data_path)

    predictor = PropertyPredictor(model_path, scaler_path)
    df_with_predictions = predictor.batch_predict(df)

    if output_path:
        logger.info(f"Saving predictions to {output_path}")
        df_with_predictions.to_csv(output_path, index=False)

    return df_with_predictions


def predict_from_features(
    property_m2: float,
    unit_level: int,
    district: str,
    property_type: str,
    freehold: int,
    transit: int,
    market_cluster_id: str,
    **amenity_features
) -> float:
    """
    Predict price from individual feature values.

    Args:
        property_m2: Property size in square meters
        unit_level: Floor level
        district: District name
        property_type: Property type
        freehold: Freehold flag (0 or 1)
        transit: Transit accessibility flag (0 or 1)
        market_cluster_id: Market cluster ID
        **amenity_features: Additional amenity features (distances, counts, etc.)

    Returns:
        float: Predicted price in RM/m²

    Example:
        >>> price = predict_from_features(
        ...     property_m2=1000,
        ...     unit_level=25,
        ...     district='kuala lumpur',
        ...     property_type='condominium/apartment',
        ...     freehold=1,
        ...     transit=1,
        ...     market_cluster_id='KL_017',
        ...     walk_dist_to_rail_station=0.4,
        ...     mall_count=5,
        ...     # ... other features
        ... )
    """
    # Combine all features
    features = {
        'property_m2': property_m2,
        'unit_level': unit_level,
        'district': district,
        'property_type': property_type,
        'freehold': freehold,
        'transit': transit,
        'market_cluster_id': market_cluster_id,
        **amenity_features
    }

    predictor = PropertyPredictor()
    return predictor.predict_single(features)
