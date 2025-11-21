"""
Data loading and saving utilities.

Handles reading from various sources (Excel, CSV) and saving intermediate outputs.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from ..config import (
    DATA_RAW_DIR,
    DATA_INTERIM_DIR,
    DATA_PROCESSED_DIR,
    GEOCODED_DATA,
    WITH_FEATURES_DATA,
    MERGED_DATA,
    CLUSTERED_DATA,
    PROCESSED_DATA,
    PROJECT_ROOT
)

logger = logging.getLogger(__name__)

# Security limits
MAX_FILE_SIZE_MB = 500  # Maximum file size in megabytes
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Allowed directories for data loading (relative to project root)
ALLOWED_DATA_DIRS = [
    DATA_RAW_DIR,
    DATA_INTERIM_DIR,
    DATA_PROCESSED_DIR,
]


class DataSecurityError(Exception):
    """Raised when data loading security checks fail."""
    pass


def _validate_file_path(file_path: Path, check_allowed_dirs: bool = True) -> None:
    """
    Validate file path for security.

    Args:
        file_path: Path to validate
        check_allowed_dirs: If True, enforce allowed directories

    Raises:
        DataSecurityError: If path validation fails
    """
    resolved_path = file_path.resolve()

    # Check for path traversal attempts
    try:
        # Ensure path doesn't escape project root
        resolved_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        raise DataSecurityError(
            f"Path traversal detected: {file_path} resolves outside project root. "
            f"All data files must be within {PROJECT_ROOT}"
        )

    # Check against allowed directories (optional, for stricter security)
    if check_allowed_dirs:
        in_allowed_dir = any(
            str(resolved_path).startswith(str(allowed_dir.resolve()))
            for allowed_dir in ALLOWED_DATA_DIRS
        )
        if not in_allowed_dir:
            logger.warning(
                f"File {file_path} is not in standard data directories. "
                f"Consider moving to: {[str(d) for d in ALLOWED_DATA_DIRS]}"
            )


def _validate_file_size(file_path: Path, max_size_bytes: int = MAX_FILE_SIZE_BYTES) -> None:
    """
    Validate file size to prevent memory exhaustion.

    Args:
        file_path: Path to file
        max_size_bytes: Maximum allowed file size

    Raises:
        DataSecurityError: If file exceeds size limit
    """
    file_size = file_path.stat().st_size

    if file_size > max_size_bytes:
        size_mb = file_size / (1024 * 1024)
        max_mb = max_size_bytes / (1024 * 1024)
        raise DataSecurityError(
            f"File size ({size_mb:.1f} MB) exceeds limit ({max_mb:.1f} MB). "
            f"For large datasets, consider chunked loading or data sampling."
        )

    logger.debug(f"File size validation passed: {file_size / (1024*1024):.1f} MB")


def load_raw_data(
    file_path: Union[str, Path],
    file_type: str = 'auto',
    validate_path: bool = True,
    max_size_mb: Optional[float] = None
) -> pd.DataFrame:
    """
    Load raw property data from file with security validation.

    Args:
        file_path: Path to the data file
        file_type: File type ('excel', 'csv', or 'auto' to detect from extension)
        validate_path: If True, validate path is within allowed directories
        max_size_mb: Maximum file size in MB (defaults to MAX_FILE_SIZE_MB)

    Returns:
        pd.DataFrame: Loaded property data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type is not supported
        DataSecurityError: If security validation fails
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Security validations
    if validate_path:
        _validate_file_path(file_path, check_allowed_dirs=False)

    max_bytes = int(max_size_mb * 1024 * 1024) if max_size_mb else MAX_FILE_SIZE_BYTES
    _validate_file_size(file_path, max_bytes)

    # Auto-detect file type from extension
    if file_type == 'auto':
        suffix = file_path.suffix.lower()
        if suffix in ['.xlsx', '.xls']:
            file_type = 'excel'
        elif suffix == '.csv':
            file_type = 'csv'
        else:
            raise ValueError(f"Cannot auto-detect file type for: {suffix}")

    logger.info(f"Loading data from {file_path}")

    # Load based on file type
    if file_type == 'excel':
        df = pd.read_excel(file_path)
    elif file_type == 'csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    logger.info(f"Loaded {len(df)} rows × {len(df.columns)} columns")

    return df


def save_interim_data(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    stage_name: str = "interim"
) -> None:
    """
    Save interim data to CSV.

    Args:
        df: DataFrame to save
        output_path: Output file path
        stage_name: Name of pipeline stage (for logging)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {stage_name} data to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} rows × {len(df.columns)} columns")


def load_interim_data(
    stage: str,
    custom_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Load interim data from a specific pipeline stage.

    Args:
        stage: Pipeline stage ('geocoded', 'with_features', 'merged', 'clustered', 'processed')
        custom_path: Optional custom path to load from

    Returns:
        pd.DataFrame: Loaded interim data

    Raises:
        ValueError: If stage is not recognized
        FileNotFoundError: If interim file doesn't exist
    """
    # Map stage names to file paths
    stage_paths = {
        'geocoded': GEOCODED_DATA,
        'with_features': WITH_FEATURES_DATA,
        'merged': MERGED_DATA,
        'clustered': CLUSTERED_DATA,
        'processed': PROCESSED_DATA
    }

    if custom_path:
        file_path = Path(custom_path)
    elif stage in stage_paths:
        file_path = stage_paths[stage]
    else:
        raise ValueError(
            f"Unknown stage: {stage}. "
            f"Must be one of: {', '.join(stage_paths.keys())}"
        )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Interim data not found for stage '{stage}' at {file_path}. "
            f"Run previous pipeline stages first."
        )

    logger.info(f"Loading {stage} data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows × {len(df.columns)} columns")

    return df


def load_property_data_for_training(
    data_path: Optional[Union[str, Path]] = None,
    use_clustered: bool = True
) -> pd.DataFrame:
    """
    Load property data ready for model training.

    Args:
        data_path: Optional custom path to data file
        use_clustered: If True, load clustered data; otherwise load merged data

    Returns:
        pd.DataFrame: Property data ready for preprocessing
    """
    if data_path:
        df = load_raw_data(data_path)
    else:
        stage = 'clustered' if use_clustered else 'merged'
        df = load_interim_data(stage)

    logger.info(f"Loaded property data: {df.shape}")
    return df


def save_processed_data(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Save final processed data ready for model training.

    Args:
        df: Processed DataFrame
        output_path: Optional custom output path (defaults to config.PROCESSED_DATA)
    """
    if output_path is None:
        output_path = PROCESSED_DATA

    save_interim_data(df, output_path, stage_name="processed")


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics about the dataset.

    Args:
        df: DataFrame to summarize

    Returns:
        dict: Summary statistics
    """
    summary = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }

    return summary
