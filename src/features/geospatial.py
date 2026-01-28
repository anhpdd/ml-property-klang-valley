"""
Geospatial feature extraction using OpenStreetMap data.

Extracts location-based features including distances to amenities,
amenity counts, and transit ridership data.
"""

import logging
from typing import List, Optional, Dict, Tuple, Any

import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import Point
from tqdm import tqdm

from ..config import (
    OSM_AMENITY_TAGS,
    AMENITY_SEARCH_RADIUS_KM,
    DISTANCE_COLS,
    DISTRICT_OSM_IDS
)

logger = logging.getLogger(__name__)


def extract_amenity_features(
    df: pd.DataFrame,
    amenity_gdf: Optional[gpd.GeoDataFrame] = None,
    radius_km: float = AMENITY_SEARCH_RADIUS_KM
) -> pd.DataFrame:
    """
    Extract amenity-based features for all properties.
    Includes counts within radius and distances (placeholder for full network calculation).

    Args:
        df: DataFrame with property locations (must have latitude/longitude or geometry)
        amenity_gdf: GeoDataFrame with amenities. If None, queries OSM.
        radius_km: Search radius for counting amenities.

    Returns:
        pd.DataFrame: DataFrame with amenity features added
    """
    if 'geometry' not in df.columns and 'latitude' in df.columns and 'longitude' in df.columns:
        df = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326"
        )
    elif not isinstance(df, gpd.GeoDataFrame):
        logger.error(
            "Input DataFrame must be a GeoDataFrame or have lat/lon columns")
        return df

    if amenity_gdf is None:
        logger.info("Amenity GeoDataFrame not provided, fetching from OSM...")
        pois = get_amenities_from_osm(DISTRICT_OSM_IDS, OSM_AMENITY_TAGS)
        amenity_df = pd.DataFrame(pois)
        amenity_df['geometry'] = amenity_df['geometry_coords'].apply(
            lambda x: Point(x[1], x[0]) if isinstance(x, tuple) else None
        )
        amenity_gdf = gpd.GeoDataFrame(amenity_df.dropna(
            subset=['geometry']), crs="EPSG:4326")
        amenity_gdf = amenity_gdf.rename(columns={'category': 'feature_type'})

    logger.info(
        f"Extracting features for {len(df)} properties using {len(amenity_gdf)} amenities")

    # 1. Count amenities within radius
    df = count_amenities_within_radius(df, amenity_gdf, radius_km)

    # 2. Calculate nearest amenity distances (Haversine as baseline)
    for amenity_type in amenity_gdf['feature_type'].str.lower().unique():
        logger.info(f"Calculating distances to nearest {amenity_type}...")
        type_amenities = amenity_gdf[amenity_gdf['feature_type'].str.lower(
        ) == amenity_type]
        col_name = f'dist_to_{amenity_type}'

        # Simple nearest neighbor distance (Haversine fallback)
        df[col_name] = df.geometry.apply(
            lambda x: type_amenities.distance(
                x).min() * 111.32  # Rough km conversion for 4326
        )

    return df


def calculate_distances(
    property_gdf: gpd.GeoDataFrame,
    amenity_gdf: gpd.GeoDataFrame,
    G: nx.MultiDiGraph,
    network_type: str = 'walk',
    max_distance_km: float = 21.0
) -> gpd.GeoDataFrame:
    """
    Calculate travel distances along a road network from properties to nearest amenities.

    Args:
        property_gdf: GeoDataFrame of properties
        amenity_gdf: GeoDataFrame of amenities
        G: NetworkX graph of the road network
        network_type: Type of network ('walk' or 'drive')
        max_distance_km: Maximum search distance in km

    Returns:
        gpd.GeoDataFrame: Updated GeoDataFrame with distance columns
    """
    logger.info(f"Calculating {network_type} distances using network graph...")

    # CRS alignment
    graph_crs = G.graph.get('crs', 'EPSG:4326')
    if property_gdf.crs != graph_crs:
        property_gdf = property_gdf.to_crs(graph_crs)
    if amenity_gdf.crs != graph_crs:
        amenity_gdf = amenity_gdf.to_crs(graph_crs)

    result_gdf = property_gdf.copy()
    cutoff_meters = max_distance_km * 1000

    # Map to nearest nodes
    logger.info("Mapping properties and amenities to network nodes...")
    prop_nodes = ox.nearest_nodes(
        G,
        result_gdf.geometry.x.to_list(),
        result_gdf.geometry.y.to_list()
    )

    amenity_nodes = ox.nearest_nodes(
        G,
        amenity_gdf.geometry.x.to_list(),
        amenity_gdf.geometry.y.to_list()
    )

    amenity_nodes_series = pd.Series(amenity_nodes, index=amenity_gdf.index)

    # Create amenity type lookup
    amenity_type_nodes = {}
    for amenity_type in amenity_gdf['feature_type'].dropna().unique():
        type_name = str(amenity_type).lower().replace(' ', '_')
        amenity_indices = amenity_gdf[amenity_gdf['feature_type']
                                      == amenity_type].index
        amenity_type_nodes[type_name] = set(
            amenity_nodes_series.loc[amenity_indices])

        col_name = f"{network_type}_dist_to_{type_name}" if network_type == 'walk' else f"dist_to_{type_name}"
        result_gdf[col_name] = np.nan

    # Process each property
    logger.info("Calculating shortest paths...")
    for i, prop_node in enumerate(tqdm(prop_nodes, desc=f"{network_type.capitalize()} distances")):
        try:
            distances = nx.single_source_dijkstra_path_length(
                G, prop_node, cutoff=cutoff_meters, weight='length'
            )

            for type_name, target_nodes in amenity_type_nodes.items():
                col_name = f"{network_type}_dist_to_{type_name}" if network_type == 'walk' else f"dist_to_{type_name}"

                min_dist = min(
                    (distances[node]
                     for node in target_nodes if node in distances),
                    default=np.nan
                )

                result_gdf.iloc[i, result_gdf.columns.get_loc(
                    col_name)] = min_dist / 1000
        except Exception as e:
            continue

    return result_gdf


def count_amenities_within_radius(
    properties_gdf: gpd.GeoDataFrame,
    amenities_gdf: gpd.GeoDataFrame,
    radius_km: float = AMENITY_SEARCH_RADIUS_KM,
    amenity_type_col: str = 'feature_type'
) -> gpd.GeoDataFrame:
    """
    Count number of amenities of each type within a radius of each property.

    Args:
        properties_gdf: GeoDataFrame with property locations
        amenities_gdf: GeoDataFrame with amenity locations
        radius_km: Search radius in kilometers
        amenity_type_col: Column name for amenity type

    Returns:
        gpd.GeoDataFrame: Properties with count columns added
    """
    logger.info(f"Counting amenities within {radius_km}km radius...")

    result_gdf = properties_gdf.copy()

    # Ensure matching projected CRS for accurate buffering
    # Using EPSG:3857 for rough metric buffering, or better a local UTM
    orig_crs = properties_gdf.crs
    properties_proj = properties_gdf.to_crs(
        epsg=32648)  # UTM 48N (Peninsular Malaysia)
    amenities_proj = amenities_gdf.to_crs(epsg=32648)

    # Buffer properties
    radius_meters = radius_km * 1000
    properties_proj['buffered_geom'] = properties_proj.geometry.buffer(
        radius_meters)

    # Set buffered geometry as active for join
    properties_proj = properties_proj.set_geometry('buffered_geom')

    # Spatial join
    joined = gpd.sjoin(
        amenities_proj[[amenity_type_col, 'geometry']],
        properties_proj[['buffered_geom']],
        how='inner',
        predicate='within'
    )

    # Group and count
    if not joined.empty:
        counts = joined.groupby(
            [joined.index_right, amenity_type_col]).size().unstack(fill_value=0)

        for amenity_type in counts.columns:
            col_name = f"{str(amenity_type).lower().replace(' ', '_')}_count"
            result_gdf[col_name] = counts[amenity_type]
            result_gdf[col_name] = result_gdf[col_name].fillna(0).astype(int)

    # Fill missing columns that might not have been found at all
    unique_types = amenities_gdf[amenity_type_col].unique()
    for amenity_type in unique_types:
        col_name = f"{str(amenity_type).lower().replace(' ', '_')}_count"
        if col_name not in result_gdf.columns:
            result_gdf[col_name] = 0

    return result_gdf


def extract_transit_ridership(
    df: pd.DataFrame,
    ridership_df: pd.DataFrame,
    amenity_gdf: gpd.GeoDataFrame,
    radius_km: float = 1.0,
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Extract transit ridership features by matching properties to nearby stations.

    Args:
        df: Property DataFrame
        ridership_df: Ridership timeseries DataFrame
        amenity_gdf: GeoDataFrame containing rail stations
        radius_km: Matching radius
        date_col: Name of date column in both dataframes

    Returns:
        pd.DataFrame: DataFrame with ridership features added
    """
    logger.info("Extracting transit ridership features...")

    if 'station_id' not in ridership_df.columns:
        ridership_df['station_id'] = ridership_df['station_name'].str.split(
            ':').str[0]

    # Standardize dates
    df[date_col] = pd.to_datetime(df[date_col])
    ridership_df[date_col] = pd.to_datetime(ridership_df[date_col])

    # Filter for rail stations
    rail_stations = amenity_gdf[amenity_gdf['feature_type'].str.lower(
    ) == 'rail station'].copy()
    if 'station_id' not in rail_stations.columns:
        # Try to extract from name if not present
        rail_stations['station_id'] = rail_stations['name'].str.extract(
            r'([A-Z]+\d+)')

    # Join ridership with stations
    stations_with_ridership = rail_stations.merge(
        ridership_df, on='station_id', how='inner'
    )

    # Spatial join properties to buffered stations
    # (Simplified implementation: uses station points and buffers them)
    stations_proj = stations_with_ridership.to_crs(epsg=32648)
    stations_proj['geometry'] = stations_proj.geometry.buffer(radius_km * 1000)

    prop_gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    ).to_crs(epsg=32648)

    # Rename columns to avoid conflicts
    stations_proj = stations_proj.rename(columns={date_col: 'ridership_date'})
    prop_gdf = prop_gdf.rename(columns={date_col: 'prop_date'})

    # Spatial join
    joined = gpd.sjoin(prop_gdf, stations_proj,
                       how='inner', predicate='within')

    # Filter for matching months
    joined = joined[
        (joined['prop_date'].dt.year == joined['ridership_date'].dt.year) &
        (joined['prop_date'].dt.month == joined['ridership_date'].dt.month)
    ]

    # Aggregate by property index
    ridership_features = joined.groupby(joined.index).agg({
        'incoming': 'sum',
        'outgoing': 'sum'
    })

    result_df = df.copy()
    result_df['incoming_ridership_within_1km'] = ridership_features['incoming']
    result_df['outgoing_ridership_within_1km'] = ridership_features['outgoing']

    result_df['incoming_ridership_within_1km'] = result_df['incoming_ridership_within_1km'].fillna(
        0)
    result_df['outgoing_ridership_within_1km'] = result_df['outgoing_ridership_within_1km'].fillna(
        0)

    return result_df


def calculate_haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculate haversine distance between two points.

    Args:
        lat1, lon1: Coordinates of point 1
        lat2, lon2: Coordinates of point 2

    Returns:
        float: Distance in kilometers
    """
    from math import radians, sin, cos, sqrt, atan2

    R = 6371.0  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def get_nearest_amenity_distance(
    prop_lat: float,
    prop_lon: float,
    amenities_df: pd.DataFrame,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude'
) -> float:
    """
    Get distance to nearest amenity from a property.
    """
    if len(amenities_df) == 0:
        return np.inf

    # Vectorized haversine would be faster but this is simple
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
    """
    logger.info(
        f"Filling missing values in {len(distance_cols)} distance columns")

    df = df.copy()

    for col in distance_cols:
        if col not in df.columns:
            continue

        # Fill with district-level max
        if 'district' in df.columns:
            district_max = df.groupby('district')[col].transform('max')
            df[col] = df[col].fillna(district_max)

        df[col] = df[col].fillna(max_distance)
        df[col] = round(df[col], 3)

    return df
