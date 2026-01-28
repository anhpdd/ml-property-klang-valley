"""
Data loader and cleaner for residential property data.
Handles loading, concatenation, and initial cleaning of property records.
"""

# ===== IMPORTS  =====
import logging
import math
import re
from pathlib import Path
from typing import List, Dict, Optional, Union

import numpy as np
import pandas as pd

# Local
from src.config import DATA_RAW_DIR, CLEANED_DATA_DIR, PROJECT_ROOT, MAX_FILE_SIZE_BYTES

# ===== CUSTOM EXCEPTIONS =====


class DataSecurityError(Exception):
    """Raised when data security validation fails."""
    pass


# ===== CONFIGURATION =====
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
YEARS = [2023, 2024, 2025]
COLUMNS_TO_DROP = ['Unit', 'Unit        ']
ROAD_PREFIXES = ['TAMAN', 'JALAN', 'LEBUH', 'LRT', 'LORONG',
                 'LINTANG', 'SOLOK', 'BANDAR', 'KAMPUNG']

# Standardization mappings
ROAD_ABBREVIATIONS = {
    r'\b-?(?:JLN|JALN|JALANG|JLAN|OFF\s+JALAN|JALANLAN|OFJLN|JA;LAN|JALAN\.|J\.?)\b': 'JALAN',
    r'\bTMN\b': 'TAMAN',
    r'\b(OFF PERSIARAN|PERSIRN|PRSRN\.?)\b': 'PERSIARAN',
    r'\bLTG\b': 'LINTANG',
    r'\bBKT\b': 'BUKIT',
    r'\bSG\b': 'SUNGAI',
    r'\bKG\b': 'KAMPUNG',
    r'\bSLK\b': 'SOLOK',
    r'\bKLN\b': 'KILANG',
    r'\bLEBOH\b': 'LEBUH',
    r'\b(?:[A-Z]\.\s*)+': 'LORONG',
}

LEVEL_MAPPING = {
    'G': 0, 'P': 0, 'LG': 0, 'UG': 0, 'MZ': 0, 'T': 0
}

MONTH_MAP = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
}


# ===== DATA LOADING FUNCTIONS =====
def _validate_file_path(file_path: Union[str, Path], check_allowed_dirs: bool = True) -> None:
    """
    Validate that file path is within project root to prevent path traversal.
    """
    file_path = Path(file_path).resolve()
    project_root = PROJECT_ROOT.resolve()

    try:
        file_path.relative_to(project_root)
    except ValueError:
        raise DataSecurityError(
            f"Path traversal detected: {file_path} is outside project root")


def _validate_file_size(file_path: Union[str, Path], max_size_bytes: int = MAX_FILE_SIZE_BYTES) -> None:
    """
    Validate that file size is within limits to prevent memory exhaustion.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    size_bytes = file_path.stat().st_size
    if size_bytes > max_size_bytes:
        raise DataSecurityError(
            f"File size ({size_bytes} bytes) exceeds limit ({max_size_bytes} bytes)"
        )


def load_raw_data(
    file_path: Union[str, Path],
    validate_path: bool = True,
    max_size_mb: Optional[float] = None
) -> pd.DataFrame:
    """
    Load raw data from CSV or Excel file with security validation.

    Args:
        file_path: Path to the data file
        validate_path: Whether to validate path against project root
        max_size_mb: Optional override for max file size in MB

    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    file_path = Path(file_path)

    if validate_path:
        _validate_file_path(file_path)

    if max_size_mb is not None:
        _validate_file_size(file_path, int(max_size_mb * 1024 * 1024))
    else:
        _validate_file_size(file_path)

    logger.info(f"Loading raw data from {file_path}")

    if file_path.suffix == '.csv':
        return pd.read_csv(file_path)
    elif file_path.suffix in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(
            f"Cannot auto-detect file type for: {file_path.suffix}")


def load_interim_data(stage_name: str) -> pd.DataFrame:
    """
    Load data from a specific pipeline stage in the interim directory.

    Args:
        stage_name: Name of the pipeline stage (e.g., 'geocoded', 'with_features')

    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    file_path = CLEANED_DATA_DIR / f"{stage_name}.csv"
    logger.info(f"Loading interim data for stage: {stage_name}")

    if not file_path.exists():
        # Try .xlsx if .csv doesn't exist
        file_path = CLEANED_DATA_DIR / f"{stage_name}.xlsx"
        if not file_path.exists():
            raise FileNotFoundError(
                f"Interim data file not found: {file_path.stem}")

    return load_raw_data(file_path)


def save_interim_data(
    df: pd.DataFrame,
    file_path: Optional[Union[str, Path]] = None,
    stage_name: Optional[str] = None
) -> None:
    """
    Save data to the interim directory.

    Args:
        df: DataFrame to save
        file_path: Exact path to save to (optional)
        stage_name: Name of the stage for default naming (optional)
    """
    if file_path:
        file_path = Path(file_path)
    elif stage_name:
        file_path = CLEANED_DATA_DIR / f"{stage_name}.csv"
    else:
        raise ValueError("Either file_path or stage_name must be provided")

    logger.info(f"Saving interim data to {file_path}")

    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.suffix == '.csv':
        df.to_csv(file_path, index=False)
    elif file_path.suffix in ['.xls', '.xlsx']:
        df.to_excel(file_path, index=False)
    else:
        df.to_csv(file_path, index=False)


def load_yearly_data(years: List[int] = YEARS) -> pd.DataFrame:
    """
    Load and concatenate residential property data for multiple years.

    Args:
        years: List of years to load (default: 2023-2025)

    Returns:
        Concatenated DataFrame with all years

    Raises:
        FileNotFoundError: If any Excel file is missing
    """
    dataframes = []

    for year in years:
        file_path = DATA_RAW_DIR / f"res_{year}.xlsx"

        if not file_path.exists():
            raise FileNotFoundError(f"Missing data file: {file_path}")

        logger.info(f"Loading {year} data from {file_path.name}")
        df = pd.read_excel(file_path).ffill(axis=0)
        df['year'] = year
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(
        f"✓ Loaded {len(combined_df):,} property records across {len(years)} years")

    return combined_df


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for a DataFrame.
    """
    return {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.apply(lambda x: str(x)).to_dict()
    }


# ===== DATA CLEANING FUNCTIONS =====
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names: lowercase, strip whitespace, replace spaces with underscores.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with cleaned column names
    """
    df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')
    df.columns = (df.columns
                  .str.strip()
                  .str.lower()
                  .str.replace(' ', '_')
                  .str.replace('/', '_'))

    return df


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and convert numeric columns by removing formatting characters.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with cleaned numeric columns
    """
    # Define cleaning rules for each column
    numeric_cols = {
        'main_floor_area': {',': '', '-': '0'},
        'land_parcel_area': {',': '', '-': '0'},
        'transaction_price': {'RM': '', ',': ''}
    }

    for col, replacements in numeric_cols.items():
        if col in df.columns:
            df[col] = (df[col]
                       .replace(replacements, regex=True)
                       .astype(float))

    return df


def swap_mismatched_areas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix cases where main floor area is incorrectly larger than land area.
    Also handles zero values in main_floor_area.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with corrected area values
    """
    df = df.copy()

    # Fill zero main floor areas with land area
    zero_mask = df['main_floor_area'] == 0
    df.loc[zero_mask, 'main_floor_area'] = df.loc[zero_mask, 'land_parcel_area']

    # Swap when main floor > land area (likely data entry error)
    swap_mask = df['main_floor_area'] > df['land_parcel_area']
    df.loc[swap_mask, ['main_floor_area', 'land_parcel_area']] = \
        df.loc[swap_mask, ['land_parcel_area', 'main_floor_area']].values

    logger.info(
        f"Swapped {swap_mask.sum():,} rows where floor area > land area")

    return df


def clean_and_process_addresses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean, standardize, and enrich address data.

    Steps:
    1. Uppercase and strip address columns
    2. Flag missing road names
    3. Derive state from district
    4. Extract road names from scheme_name_area
    5. Standardize road abbreviations

    Args:
        df: DataFrame with address columns

    Returns:
        DataFrame with cleaned and enriched address data
    """
    df = df.copy()

    # 1. Clean address columns
    address_cols = ['district', 'mukim', 'scheme_name_area', 'road_name']
    for col in [c for c in address_cols if c in df.columns]:
        df[col] = df[col].str.strip().str.upper()

    # 2. Flag originally missing road names
    df['road_name'] = df['road_name'].replace('', np.nan)
    df['unidentified_road_name'] = df['road_name'].isna()

    # 3. Derive state from district
    conditions = [
        df['district'].str.contains('KUALA LUMPUR', na=False),
        df['district'].str.contains('PUTRAJAYA', na=False)
    ]
    choices = ['WP KUALA LUMPUR', 'WP PUTRAJAYA']
    df['state'] = np.select(conditions, choices, default='Selangor')

    # 4. Extract road names from scheme_name_area
    extraction_pattern = re.compile(
        r'\b(' + '|'.join(ROAD_PREFIXES) + r')\b[^\(\)\n]*',
        flags=re.IGNORECASE
    )

    def extract_road_name(scheme_name: str) -> Optional[str]:
        """Extract road name from scheme using regex pattern."""
        if pd.isna(scheme_name):
            return None
        match = extraction_pattern.search(scheme_name)
        return match.group(0).strip() if match else None

    # Fill missing road names
    missing_mask = df['unidentified_road_name']
    df.loc[missing_mask, 'road_name'] = \
        df.loc[missing_mask, 'scheme_name_area'].apply(extract_road_name)

    # Fallback: use scheme_name_area if still missing
    df['road_name'] = df['road_name'].fillna(df['scheme_name_area'])

    # 5. Standardize abbreviations
    df['road_name'] = df['road_name'].replace(ROAD_ABBREVIATIONS, regex=True)

    return df


def clean_unit_level(unit_level_series: pd.Series) -> pd.Series:
    """
    Clean and convert unit/level strings to integers.

    Handles:
    - Standard numbers ('5', '12')
    - Ground floors ('G', 'UG', 'LG', 'P')
    - Ranges ('1-4', '2&3') - takes ceiling of average
    - Excel date errors ('1-Mar' -> '1-3')
    - Levels with letters ('3A', '13A')
    - Invalid values -> 0

    Args:
        unit_level_series: Series of unit level strings

    Returns:
        Series of cleaned integer levels
    """

    def convert_single_level(level) -> int:
        """Convert a single level value to integer."""
        # Handle non-string or empty values
        if not isinstance(level, str) or not level.strip():
            return 0

        level = level.strip().upper()

        # Check special level codes
        if level in LEVEL_MAPPING:
            return LEVEL_MAPPING[level]

        # Handle Excel date format errors (e.g., '1-MAR')
        date_match = re.match(r'^(\d+)-([A-Z]{3})$', level)
        if date_match:
            day, month_str = date_match.groups()
            if month_str in MONTH_MAP:
                return math.ceil((int(day) + MONTH_MAP[month_str]) / 2)

        # Handle ranges (e.g., '1-4', '2&3')
        range_match = re.match(r'^(\d+)\s*[-&]\s*(\d+)$', level)
        if range_match:
            start, end = map(int, range_match.groups())
            return math.ceil((start + end) / 2)

        # Handle decimals
        if '.' in level:
            try:
                return math.ceil(float(level))
            except (ValueError, TypeError):
                pass

        # Extract numeric part (handles '3A', '45718')
        numeric_match = re.match(r'^(\d+)', level)
        if numeric_match:
            num = int(numeric_match.group(1))
            # Treat huge numbers (Excel serial dates) as invalid
            return 0 if num > 1000 else num

        return 0

    return unit_level_series.apply(convert_single_level).astype(int)


# ===== MAIN PIPELINE =====
def load_and_clean_data() -> pd.DataFrame:
    """
    Main pipeline: Load and clean all property data.

    Returns:
        Cleaned DataFrame ready for analysis
    """
    logger.info("Starting data loading and cleaning pipeline...")

    # Load data
    df = load_yearly_data()

    # Clean columns
    df = clean_column_names(df)
    df = clean_numeric_columns(df)
    df = swap_mismatched_areas(df)

    # Clean addresses
    df = clean_and_process_addresses(df)

    # Clean unit levels
    df['unit_level_cleaned'] = clean_unit_level(df['unit_level'])

    logger.info(f"✓ Cleaning complete. Final shape: {df.shape}")
    logger.info(f"✓ Columns: {list(df.columns)}")

    return df


# ===== SCRIPT EXECUTION =====
if __name__ == "__main__":
    # This block only runs when script is executed directly
    master_df = load_and_clean_data()

    # Display summary statistics
    print("\n" + "="*60)
    print("UNIT LEVEL DISTRIBUTION")
    print("="*60)
    print(master_df['unit_level_cleaned'].value_counts().head(10))

    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Total records: {len(master_df):,}")
    print(f"Date range: {master_df['year'].min()} - {master_df['year'].max()}")
    print(f"Unique districts: {master_df['district'].nunique()}")
    print(
        f"Missing road names (original): {master_df['unidentified_road_name'].sum():,}")
