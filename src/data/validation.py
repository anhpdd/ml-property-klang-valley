"""
Data validation utilities.

Validates property data quality, checks for missing values,
and ensures data integrity throughout the pipeline.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from ..config import (
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_VARIABLE,
    DISTANCE_COLS
)

logger = logging.getLogger(__name__)


def validate_property_data(
    df: pd.DataFrame,
    stage: str = "raw"
) -> Dict[str, any]:
    """
    Validate property data and return validation report.

    Args:
        df: DataFrame to validate
        stage: Pipeline stage ('raw', 'geocoded', 'with_features', 'clustered')

    Returns:
        dict: Validation report with warnings and errors
    """
    logger.info(f"Validating {stage} data: {df.shape}")

    validation_report = {
        'stage': stage,
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'warnings': [],
        'errors': [],
        'passed': True
    }

    # Check for completely empty DataFrame
    if len(df) == 0:
        validation_report['errors'].append("DataFrame is empty!")
        validation_report['passed'] = False
        return validation_report

    # Check for duplicate records
    if 'records_id' in df.columns:
        n_duplicates = df['records_id'].duplicated().sum()
        if n_duplicates > 0:
            validation_report['warnings'].append(
                f"Found {n_duplicates} duplicate record IDs"
            )

    # Stage-specific validations
    if stage in ['geocoded', 'with_features', 'clustered']:
        # Should have way_id column
        if 'way_id' not in df.columns:
            validation_report['errors'].append("Missing 'way_id' column")
            validation_report['passed'] = False
        elif df['way_id'].isna().any():
            n_missing = df['way_id'].isna().sum()
            validation_report['errors'].append(
                f"{n_missing} properties missing way_id (geocoding incomplete)"
            )
            validation_report['passed'] = False

    if stage in ['with_features', 'clustered']:
        # Should have distance and count columns
        missing_features = []
        for col in DISTANCE_COLS:
            if col not in df.columns:
                missing_features.append(col)

        if missing_features:
            validation_report['errors'].append(
                f"Missing feature columns: {missing_features}"
            )
            validation_report['passed'] = False

    if stage == 'clustered':
        # Should have market_cluster_id
        if 'market_cluster_id' not in df.columns:
            validation_report['errors'].append("Missing 'market_cluster_id' column")
            validation_report['passed'] = False

    # Check target variable
    if TARGET_VARIABLE in df.columns:
        # Check for negative prices
        if (df[TARGET_VARIABLE] < 0).any():
            n_negative = (df[TARGET_VARIABLE] < 0).sum()
            validation_report['errors'].append(
                f"Found {n_negative} negative prices"
            )
            validation_report['passed'] = False

        # Check for extreme outliers (>3 std from mean)
        mean_price = df[TARGET_VARIABLE].mean()
        std_price = df[TARGET_VARIABLE].std()
        outliers = df[
            (df[TARGET_VARIABLE] > mean_price + 3 * std_price) |
            (df[TARGET_VARIABLE] < mean_price - 3 * std_price)
        ]
        if len(outliers) > 0:
            validation_report['warnings'].append(
                f"Found {len(outliers)} extreme price outliers (>3 std from mean)"
            )

    # Log validation results
    if validation_report['passed']:
        logger.info(f"✅ Validation passed for {stage} data")
    else:
        logger.error(f"❌ Validation failed for {stage} data")

    for warning in validation_report['warnings']:
        logger.warning(warning)

    for error in validation_report['errors']:
        logger.error(error)

    return validation_report


def check_missing_values(
    df: pd.DataFrame,
    critical_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Check for missing values and return summary.

    Args:
        df: DataFrame to check
        critical_columns: List of columns that should not have missing values

    Returns:
        pd.DataFrame: Missing value summary
    """
    logger.info("Checking for missing values")

    missing_summary = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum().values,
        'missing_percent': (df.isnull().sum() / len(df) * 100).values
    })

    missing_summary = missing_summary[missing_summary['missing_count'] > 0]
    missing_summary = missing_summary.sort_values('missing_count', ascending=False)

    if len(missing_summary) > 0:
        logger.warning(f"Found missing values in {len(missing_summary)} columns")
        logger.warning(f"\n{missing_summary.to_string()}")
    else:
        logger.info("No missing values found")

    # Check critical columns
    if critical_columns:
        critical_missing = missing_summary[
            missing_summary['column'].isin(critical_columns)
        ]
        if len(critical_missing) > 0:
            logger.error(
                f"Critical columns have missing values:\n{critical_missing.to_string()}"
            )

    return missing_summary


def validate_coordinates(
    df: pd.DataFrame,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude'
) -> Dict[str, any]:
    """
    Validate geographic coordinates.

    Args:
        df: DataFrame with coordinate columns
        lat_col: Latitude column name
        lon_col: Longitude column name

    Returns:
        dict: Validation report
    """
    logger.info("Validating geographic coordinates")

    report = {
        'total_properties': len(df),
        'valid_coordinates': 0,
        'invalid_coordinates': 0,
        'missing_coordinates': 0,
        'issues': []
    }

    # Check if columns exist
    if lat_col not in df.columns or lon_col not in df.columns:
        report['issues'].append(f"Missing coordinate columns: {lat_col} or {lon_col}")
        return report

    # Check for missing coordinates
    missing_lat = df[lat_col].isna()
    missing_lon = df[lon_col].isna()
    missing_any = missing_lat | missing_lon
    report['missing_coordinates'] = missing_any.sum()

    # Check for invalid ranges (Malaysia is roughly 1-7°N, 99-120°E)
    valid_lat = df[lat_col].between(1, 7)
    valid_lon = df[lon_col].between(99, 120)
    valid_coords = valid_lat & valid_lon & ~missing_any

    report['valid_coordinates'] = valid_coords.sum()
    report['invalid_coordinates'] = len(df) - report['valid_coordinates'] - report['missing_coordinates']

    if report['invalid_coordinates'] > 0:
        report['issues'].append(
            f"{report['invalid_coordinates']} properties have coordinates outside Malaysia"
        )

    if report['missing_coordinates'] > 0:
        report['issues'].append(
            f"{report['missing_coordinates']} properties missing coordinates"
        )

    logger.info(
        f"Coordinate validation: "
        f"{report['valid_coordinates']} valid, "
        f"{report['invalid_coordinates']} invalid, "
        f"{report['missing_coordinates']} missing"
    )

    return report


def validate_feature_ranges(
    df: pd.DataFrame,
    feature_ranges: Optional[Dict[str, tuple]] = None
) -> Dict[str, any]:
    """
    Validate that features are within expected ranges.

    Args:
        df: DataFrame with features
        feature_ranges: Dict mapping feature names to (min, max) tuples

    Returns:
        dict: Validation report
    """
    if feature_ranges is None:
        # Default ranges for common features
        feature_ranges = {
            'property_m2': (10, 10000),  # 10 m² to 10,000 m²
            'unit_level': (0, 100),  # Ground floor to 100th floor
            'price_m2': (100, 200000),  # RM 100/m² to RM 200,000/m²
        }

        # Distance features should be positive and < 50 km
        for col in DISTANCE_COLS:
            if col in df.columns:
                feature_ranges[col] = (0, 50)

    logger.info(f"Validating feature ranges for {len(feature_ranges)} features")

    report = {
        'total_features': len(feature_ranges),
        'passed': 0,
        'failed': 0,
        'issues': []
    }

    for feature, (min_val, max_val) in feature_ranges.items():
        if feature not in df.columns:
            continue

        out_of_range = df[
            (df[feature] < min_val) | (df[feature] > max_val)
        ]

        if len(out_of_range) > 0:
            report['failed'] += 1
            report['issues'].append(
                f"{feature}: {len(out_of_range)} values outside range "
                f"[{min_val}, {max_val}]"
            )
        else:
            report['passed'] += 1

    logger.info(
        f"Feature range validation: "
        f"{report['passed']} passed, {report['failed']} failed"
    )

    for issue in report['issues']:
        logger.warning(issue)

    return report


def validate_temporal_split(
    df: pd.DataFrame,
    year_col: str = 'year',
    split_year: int = 2025
) -> Dict[str, any]:
    """
    Validate temporal train/test split.

    Args:
        df: DataFrame with year column
        year_col: Column containing year information
        split_year: Year threshold for splitting

    Returns:
        dict: Split validation report
    """
    logger.info(f"Validating temporal split at year {split_year}")

    report = {
        'split_year': split_year,
        'pre_split_count': 0,
        'post_split_count': 0,
        'total_count': len(df),
        'pre_split_percent': 0,
        'post_split_percent': 0
    }

    if year_col not in df.columns:
        report['error'] = f"Missing year column: {year_col}"
        logger.error(report['error'])
        return report

    report['pre_split_count'] = (df[year_col] < split_year).sum()
    report['post_split_count'] = (df[year_col] >= split_year).sum()
    report['pre_split_percent'] = report['pre_split_count'] / len(df) * 100
    report['post_split_percent'] = report['post_split_count'] / len(df) * 100

    logger.info(
        f"Temporal split: "
        f"Pre-{split_year}: {report['pre_split_count']} ({report['pre_split_percent']:.1f}%), "
        f"{split_year}+: {report['post_split_count']} ({report['post_split_percent']:.1f}%)"
    )

    return report
