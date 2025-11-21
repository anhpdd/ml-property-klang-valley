"""
Data preprocessing for model training.

Handles feature scaling, encoding, transformation, and train/test splitting.
"""

import logging
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from ..config import (
    DISTANCE_COLS,
    COUNT_COLS,
    PROPERTY_COLS,
    RIDERSHIP_COLS,
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
    COLUMNS_TO_DROP,
    TARGET_VARIABLE,
    TEST_SIZE,
    RANDOM_STATE,
    TEMPORAL_SPLIT_YEAR,
    MAX_DISTANCE_KM
)

logger = logging.getLogger(__name__)


def clean_property_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw property data.

    Performs:
    - Recalculate price_m2 based on land_m2
    - Convert negative unit_level to 0
    - Drop redundant columns

    Args:
        df: Raw property DataFrame

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    logger.info(f"Cleaning property data: {df.shape}")

    df = df.copy()

    # Recalculate price_m2 based on land area (more accurate than property area)
    if 'transaction_price' in df.columns and 'land_m2' in df.columns:
        df.drop(columns='price_m2', errors='ignore', inplace=True)

        # Check for zero or negative land_m2 values to prevent division by zero
        invalid_land = (df['land_m2'] <= 0) | df['land_m2'].isna()
        invalid_count = invalid_land.sum()

        if invalid_count > 0:
            logger.warning(
                f"Found {invalid_count} properties with invalid land_m2 (<=0 or NaN). "
                f"These will have price_m2 set to NaN and should be filtered out."
            )

        # Safe division - invalid values become NaN
        df['price_m2'] = df['transaction_price'] / df['land_m2'].replace(0, np.nan)

        # Check for infinite values that might result from very small land_m2
        inf_count = np.isinf(df['price_m2']).sum()
        if inf_count > 0:
            logger.warning(
                f"Found {inf_count} infinite price_m2 values (likely from very small land_m2). "
                f"Setting these to NaN."
            )
            df.loc[np.isinf(df['price_m2']), 'price_m2'] = np.nan

        valid_count = df['price_m2'].notna().sum()
        logger.info(f"Recalculated price_m2 based on land_m2 ({valid_count} valid values)")

    # Convert negative unit levels (LG = Lower Ground) to 0 (Ground)
    if 'unit_level' in df.columns:
        negative_count = (df['unit_level'] < 0).sum()
        if negative_count > 0:
            df.loc[df['unit_level'] < 0, 'unit_level'] = 0
            logger.info(f"Converted {negative_count} negative unit_level values to 0")

    # Drop redundant columns
    columns_to_drop_actual = [col for col in COLUMNS_TO_DROP if col in df.columns]
    if columns_to_drop_actual:
        df.drop(columns=columns_to_drop_actual, inplace=True)
        logger.info(f"Dropped {len(columns_to_drop_actual)} redundant columns")

    logger.info(f"Cleaned data shape: {df.shape}")

    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in distance columns.

    Uses district-level maximum distance, capped at 21 km.

    Args:
        df: DataFrame with distance columns

    Returns:
        pd.DataFrame: DataFrame with filled values
    """
    logger.info("Filling missing values in distance columns")

    df = df.copy()

    distance_cols = [col for col in DISTANCE_COLS if col in df.columns]

    missing_before = df[distance_cols].isnull().sum().sum()
    logger.info(f"Missing values before fill: {missing_before}")

    for col in distance_cols:
        # Fill with district-level max
        district_max = df.groupby('district')[col].transform('max')
        df[col] = df[col].fillna(district_max).fillna(MAX_DISTANCE_KM)

        # Round to 3 decimal places
        df[col] = round(df[col], 3)

    missing_after = df[distance_cols].isnull().sum().sum()
    logger.info(f"Missing values after fill: {missing_after}")

    if missing_after == 0:
        logger.info("✅ All missing values filled successfully")

    return df


def create_temporal_split(
    df: pd.DataFrame,
    split_year: int = TEMPORAL_SPLIT_YEAR,
    year_col: str = 'year'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create temporal train/test split.

    Splits data by year to simulate real-world forecasting scenario.

    Args:
        df: DataFrame with year column
        split_year: Year threshold for splitting
        year_col: Column containing year information

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (pre_split_df, post_split_df)
    """
    logger.info(f"Creating temporal split at year {split_year}")

    df_pre = df[df[year_col] < split_year].copy()
    df_post = df[df[year_col] >= split_year].copy()

    logger.info(
        f"Pre-{split_year}: {len(df_pre)} samples ({len(df_pre)/len(df)*100:.1f}%)"
    )
    logger.info(
        f"{split_year}+: {len(df_post)} samples ({len(df_post)/len(df)*100:.1f}%)"
    )

    return df_pre, df_post


def create_preprocessing_pipeline(
    distance_cols: list = DISTANCE_COLS,
    count_cols: list = COUNT_COLS,
    property_cols: list = PROPERTY_COLS,
    ridership_cols: list = RIDERSHIP_COLS
) -> ColumnTransformer:
    """
    Create preprocessing pipeline for continuous features.

    Applies log transformation followed by standard scaling.

    Args:
        distance_cols: Distance feature columns
        count_cols: Count feature columns
        property_cols: Property attribute columns
        ridership_cols: Ridership feature columns

    Returns:
        ColumnTransformer: Fitted preprocessing pipeline
    """
    logger.info("Creating preprocessing pipeline")

    # Log transformation + scaling pipeline
    log_scale_pipeline = Pipeline([
        ('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
        ('scale', StandardScaler())
    ])

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('distance', log_scale_pipeline, distance_cols),
            ('counts', log_scale_pipeline, count_cols),
            ('property', log_scale_pipeline, property_cols),
            ('ridership', log_scale_pipeline, ridership_cols)
        ],
        remainder='drop'
    )

    return preprocessor


def preprocess_for_training(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    temporal_split: bool = True,
    random_state: int = RANDOM_STATE
) -> Dict[str, any]:
    """
    Complete preprocessing pipeline for model training.

    Args:
        df: Cleaned property DataFrame
        test_size: Fraction of data for testing (if temporal_split=False)
        temporal_split: If True, use temporal validation; if False, random split
        random_state: Random seed

    Returns:
        dict: Dictionary containing all preprocessed data and transformers
    """
    logger.info("Starting complete preprocessing pipeline")

    # Step 1: Separate features and target
    X = df.drop(columns=TARGET_VARIABLE)
    y = df[TARGET_VARIABLE]

    # Step 2: Create train/test split
    if temporal_split:
        df_pre2025, df_2025 = create_temporal_split(df)

        X_pre2025 = df_pre2025.drop(columns=TARGET_VARIABLE)
        y_pre2025 = df_pre2025[TARGET_VARIABLE]

        X_2025 = df_2025.drop(columns=TARGET_VARIABLE)
        y_2025 = df_2025[TARGET_VARIABLE]

        # Further split pre-2025 into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_pre2025, y_pre2025,
            test_size=test_size,
            random_state=random_state
        )

        # Use 2025 as validation holdout
        X_val, y_val = X_2025, y_2025

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )
        X_val, y_val = None, None

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    if X_val is not None:
        logger.info(f"Validation set (2025 holdout): {len(X_val)} samples")

    # Step 3: Create and fit transformers on TRAINING data only
    # Continuous features
    continuous_cols = [col for col in CONTINUOUS_FEATURES if col in X_train.columns]
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(X_train[continuous_cols])

    # Categorical features
    categorical_cols = [col for col in CATEGORICAL_FEATURES if col in X_train.columns]
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoder.fit(X_train[categorical_cols])

    # Step 4: Transform all datasets
    def transform_features(X, y=None):
        """Apply fitted preprocessing to dataset"""
        X_continuous = preprocessor.transform(X[continuous_cols])
        X_categorical = encoder.transform(X[categorical_cols])
        X_transformed = np.hstack([X_continuous, X_categorical])

        y_transformed = np.log1p(y) if y is not None else None

        return X_transformed, y_transformed

    X_train_scaled, y_train_log = transform_features(X_train, y_train)
    X_test_scaled, y_test_log = transform_features(X_test, y_test)

    if X_val is not None:
        X_val_scaled, y_val_log = transform_features(X_val, y_val)
    else:
        X_val_scaled, y_val_log = None, None

    # Step 5: Get feature names
    continuous_feature_names = preprocessor.get_feature_names_out().tolist()
    categorical_feature_names = encoder.get_feature_names_out(categorical_cols).tolist()
    all_feature_names = continuous_feature_names + categorical_feature_names

    logger.info(f"✅ Preprocessing complete: {len(all_feature_names)} total features")

    # Return everything
    result = {
        # Scaled features
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'X_val': X_val_scaled,
        # Log-transformed target
        'y_train_log': y_train_log,
        'y_test_log': y_test_log,
        'y_val_log': y_val_log,
        # Original target (for MAPE calculation)
        'y_train': y_train,
        'y_test': y_test,
        'y_val': y_val,
        # Transformers
        'preprocessor': preprocessor,
        'encoder': encoder,
        # Feature names
        'feature_names': all_feature_names,
        'continuous_features': continuous_cols,
        'categorical_features': categorical_cols,
        # Metadata
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_val': len(X_val) if X_val is not None else 0,
        'n_features': len(all_feature_names)
    }

    return result


def inverse_transform_predictions(
    y_pred_log: np.ndarray
) -> np.ndarray:
    """
    Inverse transform log-space predictions to original scale (RM/m²).

    Args:
        y_pred_log: Predictions in log-space

    Returns:
        np.ndarray: Predictions in original scale
    """
    return np.expm1(y_pred_log)
