"""
Geocoding utilities for converting addresses to coordinates.

Handles OpenStreetMap API interactions for geocoding property addresses
and validating coordinates against district boundaries. Includes checkpoint
system for resuming interrupted geocoding sessions.
"""

# ===== IMPORTS =====
# Standard library
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import json

# Third-party core
import pandas as pd
import numpy as np

# Geospatial

# API & Rate Limiting
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Utilities
from tqdm.auto import tqdm

# Local
from src.config import CLEANED_DATA_DIR, SUPPORT_DATA_DIR


# ===== CONFIGURATION =====
@dataclass
class GeocodingConfig:
    """Configuration for geocoding operations."""

    user_agent: str = "selangor_road_wayid_extractor_v3"
    min_delay_seconds: float = 1.1
    error_wait_seconds: float = 10.0
    max_retries: int = 2
    timeout: int = 10
    api_limit: int = 5
    checkpoint_interval: int = 100  # Save progress every N roads

    # Selangor bounding box [lat_min, lon_min], [lat_max, lon_max]
    bbox: List[List[float]] = None

    def __post_init__(self):
        if self.bbox is None:
            self.bbox = [[2.553, 100.841], [3.716, 102.39]]


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===== GEOCODING FUNCTIONS =====
def setup_geocoder(config: GeocodingConfig) -> Tuple[Nominatim, RateLimiter]:
    """
    Initialize Nominatim geocoder with rate limiting.

    Args:
        config: Geocoding configuration object

    Returns:
        Tuple of (geolocator, rate_limited_geocode_function)
    """
    geolocator = Nominatim(user_agent=config.user_agent)

    geocode = RateLimiter(
        geolocator.geocode,
        min_delay_seconds=config.min_delay_seconds,
        error_wait_seconds=config.error_wait_seconds,
        max_retries=config.max_retries
    )

    logger.info(
        f"✓ Geocoder initialized with {config.min_delay_seconds}s rate limit")

    return geolocator, geocode


def load_checkpoint(checkpoint_file: Path) -> Dict:
    """
    Load existing checkpoint data if available.

    Args:
        checkpoint_file: Path to checkpoint JSON file

    Returns:
        Dictionary with road_to_way_id and road_to_node_info
    """
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        logger.info(
            f"✓ Loaded checkpoint with {len(data['road_to_way_id'])} cached roads")
        return data

    return {
        'road_to_way_id': {},
        'road_to_node_info': {}
    }


def save_checkpoint(
    checkpoint_file: Path,
    road_to_way_id: Dict,
    road_to_node_info: Dict
) -> None:
    """
    Save current geocoding progress to checkpoint file.

    Args:
        checkpoint_file: Path to save checkpoint
        road_to_way_id: Dictionary mapping road names to way IDs
        road_to_node_info: Dictionary mapping road names to node info tuples
    """
    # Convert tuples to lists for JSON serialization
    node_info_serializable = {
        k: list(v) if isinstance(v, tuple) else v
        for k, v in road_to_node_info.items()
    }

    checkpoint_data = {
        'road_to_way_id': road_to_way_id,
        'road_to_node_info': node_info_serializable
    }

    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

    logger.info(f"✓ Checkpoint saved: {len(road_to_way_id)} roads geocoded")


def geocode_single_road(
    road_name: str,
    geocode_func: RateLimiter,
    config: GeocodingConfig
) -> Tuple[Optional[int], Tuple[Optional[int], Optional[str]]]:
    """
    Geocode a single road name using Nominatim API.

    Args:
        road_name: Name of the road to geocode
        geocode_func: Rate-limited geocoding function
        config: Geocoding configuration

    Returns:
        Tuple of (way_id, (node_id, node_address))
    """
    way_id = None
    node_info = (None, None)

    try:
        locations = geocode_func(
            f"{road_name}, Malaysia",
            timeout=config.timeout,
            viewbox=config.bbox,
            bounded=True,
            exactly_one=False,
            limit=config.api_limit
        )

        if not locations:
            return way_id, node_info

        # Prefer 'way' type results (actual road geometries)
        for loc in locations:
            if loc.raw.get('osm_type') == 'way':
                way_id = loc.raw.get('osm_id')
                break
        else:
            # Fallback to 'node' if no way found
            top_result = locations[0]
            if top_result.raw.get('osm_type') == 'node':
                node_info = (
                    top_result.raw.get('osm_id'),
                    top_result.address
                )

    except Exception as e:
        logger.warning(f"Error geocoding '{road_name}': {e}")

    return way_id, node_info


def geocode_unique_roads(
    unique_roads: np.ndarray,
    config: GeocodingConfig,
    checkpoint_file: Optional[Path] = None,
    resume: bool = True
) -> Tuple[Dict[str, int], Dict[str, Tuple]]:
    """
    Geocode all unique road names with checkpoint support.

    Args:
        unique_roads: Array of unique road names to geocode
        config: Geocoding configuration
        checkpoint_file: Path to checkpoint file (optional)
        resume: Whether to resume from checkpoint if available

    Returns:
        Tuple of (road_to_way_id, road_to_node_info) dictionaries
    """
    # Setup geocoder
    _, geocode_func = setup_geocoder(config)

    # Load checkpoint if resuming
    if resume and checkpoint_file and checkpoint_file.exists():
        checkpoint_data = load_checkpoint(checkpoint_file)
        road_to_way_id = checkpoint_data['road_to_way_id']
        road_to_node_info = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in checkpoint_data['road_to_node_info'].items()
        }
    else:
        road_to_way_id = {}
        road_to_node_info = {}

    logger.info(f"📍 Geocoding {len(unique_roads):,} unique road names...")

    # Filter out already geocoded roads
    roads_to_process = [r for r in unique_roads if r not in road_to_way_id]
    logger.info(
        f"   • Already cached: {len(unique_roads) - len(roads_to_process):,}")
    logger.info(f"   • Remaining: {len(roads_to_process):,}")

    # Geocode with progress bar
    for idx, road_name in enumerate(tqdm(roads_to_process, desc="Geocoding")):
        way_id, node_info = geocode_single_road(
            road_name, geocode_func, config)

        road_to_way_id[road_name] = way_id
        road_to_node_info[road_name] = node_info

        # Save checkpoint periodically
        if checkpoint_file and (idx + 1) % config.checkpoint_interval == 0:
            save_checkpoint(checkpoint_file, road_to_way_id, road_to_node_info)

    # Final checkpoint save
    if checkpoint_file:
        save_checkpoint(checkpoint_file, road_to_way_id, road_to_node_info)

    logger.info("✅ Geocoding complete")

    return road_to_way_id, road_to_node_info


def enrich_dataframe_with_geocodes(
    df: pd.DataFrame,
    road_to_way_id: Dict[str, int],
    road_to_node_info: Dict[str, Tuple]
) -> pd.DataFrame:
    """
    Add geocoding results to the property DataFrame.

    Args:
        df: Property DataFrame with 'road_name' column
        road_to_way_id: Mapping of road names to way IDs
        road_to_node_info: Mapping of road names to node info

    Returns:
        DataFrame with added geocoding columns
    """
    df = df.copy()

    # Add way_id column
    df['way_id'] = df['road_name'].map(road_to_way_id)

    # Add node info columns
    node_info = df['road_name'].map(road_to_node_info)
    df['found_node_id'] = node_info.apply(lambda x: x[0] if x else None)
    df['found_node_name'] = node_info.apply(lambda x: x[1] if x else None)

    return df


def calculate_geocoding_stats(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate summary statistics for geocoding results.

    Args:
        df: DataFrame with geocoding columns

    Returns:
        Dictionary with statistics
    """
    total = len(df)
    way_found = df['way_id'].notna().sum()
    node_found = df['found_node_id'].notna().sum()
    not_found = total - way_found - node_found

    return {
        'total': total,
        'way_found': way_found,
        'way_pct': way_found / total * 100,
        'node_found': node_found,
        'node_pct': node_found / total * 100,
        'not_found': not_found,
        'not_found_pct': not_found / total * 100
    }


def save_geocoding_results(
    df: pd.DataFrame,
    output_file: Path,
    columns: Optional[List[str]] = None
) -> None:
    """
    Save unique geocoding results to CSV.

    Args:
        df: DataFrame with geocoding results
        output_file: Path to output CSV file
        columns: Specific columns to save (optional)
    """
    if columns is None:
        columns = ['road_name', 'way_id', 'found_node_id',
                   'found_node_name', 'district']

    unique_results = df[columns].drop_duplicates().reset_index(drop=True)
    unique_results.to_csv(output_file, index=False)

    logger.info(
        f"✅ Saved {len(unique_results):,} unique results to '{output_file.name}'")


# ===== MAIN PIPELINE =====
def run_geocoding_pipeline(
    input_file: Path = CLEANED_DATA_DIR / "df_v1.csv",
    output_file: Path = SUPPORT_DATA_DIR / "geocoded_unique_road_names.csv",
    checkpoint_file: Path = SUPPORT_DATA_DIR / "geocoding_checkpoint.json",
    config: Optional[GeocodingConfig] = None,
    resume: bool = True
) -> pd.DataFrame:
    """
    Complete geocoding pipeline from loading data to saving results.

    Args:
        input_file: Path to input CSV with property data
        output_file: Path to save unique geocoding results
        checkpoint_file: Path for checkpoint file
        config: Geocoding configuration (uses default if None)
        resume: Whether to resume from checkpoint

    Returns:
        DataFrame with geocoding results added
    """
    if config is None:
        config = GeocodingConfig()

    logger.info("="*60)
    logger.info("GEOCODING PIPELINE STARTED")
    logger.info("="*60)

    # 1. Load data
    logger.info(f"📂 Loading data from '{input_file.name}'...")
    df = pd.read_csv(input_file)
    logger.info(f"   • Loaded {len(df):,} property records")

    # 2. Get unique roads
    unique_roads = df['road_name'].dropna().unique()
    logger.info(f"   • Found {len(unique_roads):,} unique road names")

    # 3. Geocode roads
    road_to_way_id, road_to_node_info = geocode_unique_roads(
        unique_roads,
        config,
        checkpoint_file,
        resume
    )

    # 4. Enrich DataFrame
    logger.info("🔗 Adding geocoding results to DataFrame...")
    df = enrich_dataframe_with_geocodes(df, road_to_way_id, road_to_node_info)

    # 5. Save unique results
    save_geocoding_results(df, output_file)

    # 6. Display statistics
    stats = calculate_geocoding_stats(df)
    logger.info("\n" + "="*60)
    logger.info("📊 GEOCODING SUMMARY")
    logger.info("="*60)
    logger.info(
        f"   • Way IDs found:  {stats['way_found']:,} ({stats['way_pct']:.1f}%)")
    logger.info(
        f"   • Node fallbacks: {stats['node_found']:,} ({stats['node_pct']:.1f}%)")
    logger.info(
        f"   • Not found:      {stats['not_found']:,} ({stats['not_found_pct']:.1f}%)")
    logger.info("="*60)

    return df


# ===== SCRIPT EXECUTION =====
if __name__ == "__main__":
    # Run the complete pipeline
    enriched_df = run_geocoding_pipeline()

    # Optionally save the full enriched dataset
    full_output = CLEANED_DATA_DIR / "df_v1_geocoded.csv"
    enriched_df.to_csv(full_output, index=False)
    logger.info(f"💾 Full dataset saved to '{full_output.name}'")
