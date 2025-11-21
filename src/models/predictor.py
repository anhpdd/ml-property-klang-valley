"""
Property price prediction interface.

Provides a high-level API for making predictions on new property data.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pickle

from ..config import PRODUCTION_MODEL, SCALER_MODEL, MODELS_DIR, MODEL_METADATA

logger = logging.getLogger(__name__)

# Expected number of features for production model
EXPECTED_FEATURE_COUNT = 279

# Trusted model hashes (SHA256) - update after training new models
# Set to None to skip hash verification (development only)
TRUSTED_MODEL_HASHES: Optional[Dict[str, str]] = None  # TODO: Add hashes after training


def _verify_file_hash(file_path: Path, expected_hash: Optional[str] = None) -> bool:
    """
    Verify file integrity using SHA256 hash.

    Args:
        file_path: Path to file to verify
        expected_hash: Expected SHA256 hash (hex string)

    Returns:
        bool: True if hash matches or verification skipped

    Raises:
        SecurityError: If hash doesn't match
    """
    if expected_hash is None:
        logger.warning(f"Hash verification skipped for {file_path.name} - not recommended for production")
        return True

    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)

    actual_hash = sha256.hexdigest()

    if actual_hash != expected_hash:
        raise SecurityError(
            f"Model file integrity check failed for {file_path.name}. "
            f"Expected hash: {expected_hash[:16]}..., got: {actual_hash[:16]}... "
            f"The file may have been tampered with."
        )

    logger.info(f"Hash verification passed for {file_path.name}")
    return True


class SecurityError(Exception):
    """Raised when security validation fails."""
    pass


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

    def load_model(self, verify_hash: bool = True) -> None:
        """
        Load trained model and scaler from disk with security validation.

        Args:
            verify_hash: If True, verify file integrity before loading

        Raises:
            FileNotFoundError: If model/scaler files don't exist
            SecurityError: If hash verification fails

        Warning:
            pickle.load() can execute arbitrary code. Only load models from
            trusted sources. Consider using skops.io for enhanced security.
        """
        logger.info(f"Loading model from {self.model_path}")

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                f"Train a model first using notebooks/5_Modelling.ipynb"
            )

        # Validate model file is within expected directory (prevent path traversal)
        model_resolved = self.model_path.resolve()
        if not str(model_resolved).startswith(str(MODELS_DIR.resolve())):
            raise SecurityError(
                f"Model file must be within {MODELS_DIR}. "
                f"Got: {model_resolved}"
            )

        # Verify file integrity if enabled and hashes are configured
        if verify_hash and TRUSTED_MODEL_HASHES:
            model_hash = TRUSTED_MODEL_HASHES.get(self.model_path.name)
            _verify_file_hash(self.model_path, model_hash)

        # Load model with security warning
        logger.warning(
            "Loading pickle file - ensure this file is from a trusted source. "
            "Malicious pickle files can execute arbitrary code."
        )
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

        logger.info(f"Loading scaler from {self.scaler_path}")

        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler file not found: {self.scaler_path}. "
                f"Train a model first using notebooks/5_Modelling.ipynb"
            )

        # Validate scaler file path
        scaler_resolved = self.scaler_path.resolve()
        if not str(scaler_resolved).startswith(str(MODELS_DIR.resolve())):
            raise SecurityError(
                f"Scaler file must be within {MODELS_DIR}. "
                f"Got: {scaler_resolved}"
            )

        # Verify scaler integrity
        if verify_hash and TRUSTED_MODEL_HASHES:
            scaler_hash = TRUSTED_MODEL_HASHES.get(self.scaler_path.name)
            _verify_file_hash(self.scaler_path, scaler_hash)

        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        logger.info("✅ Model and scaler loaded successfully")

    def _validate_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        expected_features: Optional[List[str]] = None
    ) -> None:
        """
        Validate input features before prediction.

        Args:
            X: Input features
            expected_features: Optional list of expected feature names

        Raises:
            ValueError: If feature count doesn't match or features are invalid
        """
        # Get feature count
        n_features = X.shape[1] if len(X.shape) > 1 else 1

        if n_features != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Feature count mismatch: expected {EXPECTED_FEATURE_COUNT} features, "
                f"got {n_features}. Ensure data is preprocessed correctly using "
                f"the same pipeline as training."
            )

        # Check for NaN/Inf values
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        if np.any(np.isnan(X_array)):
            nan_count = np.isnan(X_array).sum()
            raise ValueError(
                f"Input contains {nan_count} NaN values. "
                f"Ensure all missing values are filled before prediction."
            )

        if np.any(np.isinf(X_array)):
            inf_count = np.isinf(X_array).sum()
            raise ValueError(
                f"Input contains {inf_count} infinite values. "
                f"Check for division by zero or invalid transformations."
            )

        logger.debug(f"Feature validation passed: {n_features} features")

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_log_space: bool = False,
        validate: bool = True
    ) -> np.ndarray:
        """
        Predict property prices.

        Args:
            X: Features (must be preprocessed with exactly 279 features)
            return_log_space: If True, return predictions in log-space
            validate: If True, validate feature count and values

        Returns:
            np.ndarray: Predicted prices (RM/m²)

        Raises:
            ValueError: If features are invalid (wrong count, NaN, Inf)
        """
        if self.model is None or self.scaler is None:
            logger.info("Model not loaded, loading now...")
            self.load_model()

        logger.info(f"Making predictions for {len(X)} properties...")

        # Validate features before prediction
        if validate:
            self._validate_features(X)

        # Make predictions in log-space
        y_pred_log = self.model.predict(X)

        if return_log_space:
            logger.info("Returning predictions in log-space")
            return y_pred_log

        # Inverse transform to original scale (RM/m²)
        y_pred = np.expm1(y_pred_log)

        # Validate predictions (prices must be positive)
        if np.any(y_pred <= 0):
            negative_count = (y_pred <= 0).sum()
            logger.warning(
                f"{negative_count} predictions are non-positive. "
                f"This may indicate issues with input data or model."
            )
            # Clip to minimum reasonable price
            y_pred = np.maximum(y_pred, 1.0)

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
