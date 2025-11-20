"""
Geospatial feature extraction using OpenStreetMap data.

Extracts location-based features including distances to amenities,
amenity counts, and transit ridership data.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

from ..config import (
    OSM_AMENITY_TAGS,
    AMENITY_SEARCH_RADIUS_KM,
    OSM_API_RATE_LIMIT,
    OSM_MAX_RETRIES,
    OSM_RETRY_BACKOFF,
    DISTANCE_COLS,
    COUNT_COLS
)

logger = logging.getLogger(__name__)


def extract_amenity_features(
    df: pd.DataFrame,
    amenity_types: Optional[List[str]] = None,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Extract amenity-based features for all properties.

    Args:
        df: DataFrame with property locations (must have lat/lon)
        amenity_types: List of amenity types to extract (defaults to all in config)
        show_progress: Whether to show progress bar

    Returns:
        pd.DataFrame: DataFrame with amenity features added

    Note:
        This is a high-level orchestrator. See notebooks/2_1_Amenity_OSM_search.ipynb
        for detailed implementation using osmnx.features_from_place().
    """
    if amenity_types is None:
        amenity_types = list(OSM_AMENITY_TAGS.keys())

    logger.info(f"Extracting {len(amenity_types)} amenity types for {len(df)} properties")

    df = df.copy()

    # Extract each amenity type
    for amenity in amenity_types:
        logger.info(f"Processing {amenity} amenities...")

        # TODO: Implement OSM querying
        # 1. Query OSM for amenities: ox.features_from_place(district, tags=OSM_AMENITY_TAGS[amenity])
        # 2. Calculate straight-line distances
        # 3. Calculate network walking distances
        # 4. Count amenities within radius

        # Placeholder columns
        df[f'dist_to_{amenity}'] = np.nan
        df[f'walk_dist_to_{amenity}'] = np.nan
        df[f'{amenity}_count'] = 0

        time.sleep(OSM_API_RATE_LIMIT)  # Rate limiting

    logger.info("Amenity feature extraction complete")

    return df


def calculate_distances(
    properties_df: pd.DataFrame,
    amenities_df: pd.DataFrame,
    distance_type: str = 'haversine',
    lat_col: str = 'latitude',
    lon_col: str = 'longitude'
) -> pd.DataFrame:
    """
    Calculate distances between properties and amenities.

    Args:
        properties_df: DataFrame with property locations
        amenities_df: DataFrame with amenity locations
        distance_type: Type of distance ('haversine', 'network')
        lat_col: Latitude column name
        lon_col: Longitude column name

    Returns:
        pd.DataFrame: Properties with distance column added

    Note:
        For network distances, requires road network graph from osmnx.
        See notebooks/2_1_Amenity_OSM_search.ipynb for implementation.
    """
    logger.info(
        f"Calculating {distance_type} distances for "
        f"{len(properties_df)} properties to {len(amenities_df)} amenities"
    )

    # TODO: Implement distance calculation
    if distance_type == 'haversine':
        # Use haversine formula for straight-line distance
        # distances = calculate_haversine_distances(properties_df, amenities_df)
        pass
    elif distance_type == 'network':
        # Use osmnx shortest path for walking distance
        # distances = calculate_network_distances(properties_df, amenities_df, road_graph)
        pass
    else:
        raise ValueError(f"Unknown distance type: {distance_type}")

    logger.warning("calculate_distances not fully implemented - placeholder only")

    return properties_df


def count_amenities_within_radius(
    properties_df: pd.DataFrame,
    amenities_df: pd.DataFrame,
    radius_km: float = AMENITY_SEARCH_RADIUS_KM,
    amenity_name: str = 'amenity'
) -> pd.DataFrame:
    """
    Count number of amenities within radius of each property.

    Args:
        properties_df: DataFrame with property locations
        amenities_df: DataFrame with amenity locations
        radius_km: Search radius in kilometers
        amenity_name: Name of amenity type for column naming

    Returns:
        pd.DataFrame: Properties with count column added
    """
    logger.info(
        f"Counting {amenity_name} within {radius_km}km for "
        f"{len(properties_df)} properties"
    )

    # TODO: Implement radius count
    # 1. For each property, calculate distances to all amenities
    # 2. Count amenities within radius_km
    # 3. Add count column

    properties_df = properties_df.copy()
    properties_df[f'{amenity_name}_count'] = 0

    logger.warning("count_amenities_within_radius not fully implemented - placeholder only")

    return properties_df


def extract_transit_ridership(
    df: pd.DataFrame,
    ridership_data_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract transit ridership features.

    Aggregates Rapid KL (LRT/MRT) ridership data and assigns to properties
    within 1km of stations.

    Args:
        df: DataFrame with property locations
        ridership_data_path: Path to ridership CSV data

    Returns:
        pd.DataFrame: DataFrame with ridership features added

    Note:
        See notebooks/2_2_Ridership_data_extraction.ipynb for implementation
        using Prasarana Malaysia ridership data.
    """
    logger.info(f"Extracting transit ridership for {len(df)} properties")

    df = df.copy()

    # TODO: Implement ridership extraction
    # 1. Load ridership data from Prasarana Malaysia
    # 2. Match properties to nearby stations (within 1km)
    # 3. Aggregate incoming/outgoing ridership

    # Placeholder columns
    df['incoming_ridership_within_1km'] = 0
    df['outgoing_ridership_within_1km'] = 0

    logger.warning("extract_transit_ridership not fully implemented - placeholder only")

    return df


def calculate_haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculate haversine distance between two points.

    Args:
        lat1: Latitude of point 1 (degrees)
        lon1: Longitude of point 1 (degrees)
        lat2: Latitude of point 2 (degrees)
        lon2: Longitude of point 2 (degrees)

    Returns:
        float: Distance in kilometers
    """
    from math import radians, sin, cos, sqrt, atan2

    # Earth radius in kilometers
    R = 6371.0

    # Convert to radians
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance


def get_nearest_amenity_distance(
    prop_lat: float,
    prop_lon: float,
    amenities_df: pd.DataFrame,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude'
) -> float:
    """
    Get distance to nearest amenity from a property.

    Args:
        prop_lat: Property latitude
        prop_lon: Property longitude
        amenities_df: DataFrame of amenities with coordinates
        lat_col: Latitude column in amenities_df
        lon_col: Longitude column in amenities_df

    Returns:
        float: Distance to nearest amenity in km
    """
    if len(amenities_df) == 0:
        return np.inf

    distances = amenities_df.apply(
        lambda row: calculate_haversine_distance(
            prop_lat, prop_lon,
            row[lat_col], row[lon_col]
        ),
        axis=1
    )

    return distances.min()


def fill_missing_distances(
    df: pd.DataFrame,
    distance_cols: List[str] = DISTANCE_COLS,
    max_distance: float = 21.0
) -> pd.DataFrame:
    """
    Fill missing distance values with district-level max.

    Args:
        df: DataFrame with distance columns
        distance_cols: List of distance column names
        max_distance: Cap value if district max is also missing

    Returns:
        pd.DataFrame: DataFrame with filled distances
    """
    logger.info(f"Filling missing values in {len(distance_cols)} distance columns")

    df = df.copy()

    for col in distance_cols:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            continue

        # Fill with district-level max
        district_max = df.groupby('district')[col].transform('max')
        df[col] = df[col].fillna(district_max).fillna(max_distance)

        # Round to 3 decimal places
        df[col] = round(df[col], 3)

    missing_after = df[distance_cols].isnull().sum().sum()
    if missing_after == 0:
        logger.info("✅ All distance missing values filled successfully")
    else:
        logger.warning(f"⚠️ {missing_after} missing values remain")

    return df
