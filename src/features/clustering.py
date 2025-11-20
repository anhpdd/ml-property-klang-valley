"""
Spatial clustering for market segmentation.

Consolidates 18,000+ unique road names into 238 meaningful market segments
using DBSCAN and K-Means clustering algorithms.
"""

import logging
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from ..config import (
    DBSCAN_EPS_PERCENTILE_MIN,
    DBSCAN_EPS_PERCENTILE_MAX,
    DBSCAN_MIN_SAMPLES,
    DBSCAN_METRIC,
    DBSCAN_MAX_NOISE_THRESHOLD,
    KMEANS_N_CLUSTERS,
    CLUSTER_PREFIXES,
    RANDOM_STATE
)

logger = logging.getLogger(__name__)


def cluster_market_segments(
    df: pd.DataFrame,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    road_col: str = 'road_name',
    district_col: str = 'district',
    method: str = 'auto'
) -> pd.DataFrame:
    """
    Create market clusters from property locations.

    Consolidates unique road names into spatial market segments using
    DBSCAN (density-based) or K-Means clustering.

    Args:
        df: DataFrame with property locations
        lat_col: Latitude column name
        lon_col: Longitude column name
        road_col: Road name column (for aggregation)
        district_col: District column (for cluster naming)
        method: Clustering method ('dbscan', 'kmeans', 'auto')
                'auto' tries DBSCAN first, falls back to K-Means if too much noise

    Returns:
        pd.DataFrame: DataFrame with cluster assignments added

    Note:
        See notebooks/4_Clustering.ipynb for detailed implementation.
    """
    logger.info(f"Starting market segmentation clustering for {len(df)} properties")

    df = df.copy()

    # Step 1: Calculate median coordinates for each road
    road_centroids = df.groupby(road_col).agg({
        lat_col: 'median',
        lon_col: 'median',
        district_col: 'first'
    }).reset_index()

    logger.info(f"Aggregated {len(df)} properties into {len(road_centroids)} unique roads")

    # Step 2: Cluster the road centroids
    if method == 'auto':
        # Try DBSCAN first
        road_centroids, noise_ratio = _cluster_with_dbscan(road_centroids, lat_col, lon_col)

        # Fall back to K-Means if too much noise
        if noise_ratio > DBSCAN_MAX_NOISE_THRESHOLD:
            logger.warning(
                f"DBSCAN produced {noise_ratio:.1%} noise points (threshold: {DBSCAN_MAX_NOISE_THRESHOLD:.1%}). "
                f"Falling back to K-Means."
            )
            road_centroids = _cluster_with_kmeans(road_centroids, lat_col, lon_col)
    elif method == 'dbscan':
        road_centroids, noise_ratio = _cluster_with_dbscan(road_centroids, lat_col, lon_col)
    elif method == 'kmeans':
        road_centroids = _cluster_with_kmeans(road_centroids, lat_col, lon_col)
    else:
        raise ValueError(f"Unknown clustering method: {method}. Must be 'auto', 'dbscan', or 'kmeans'")

    # Step 3: Assign cluster IDs with district prefixes
    road_centroids = assign_cluster_ids(road_centroids, district_col)

    # Step 4: Map clusters back to original properties
    df = df.merge(
        road_centroids[[road_col, 'market_cluster_id', 'market_cluster', 'clustering_method', 'is_noise']],
        on=road_col,
        how='left'
    )

    logger.info(f"✅ Clustering complete: {df['market_cluster_id'].nunique()} unique market segments created")

    return df


def _cluster_with_dbscan(
    road_centroids: pd.DataFrame,
    lat_col: str,
    lon_col: str
) -> Tuple[pd.DataFrame, float]:
    """
    Perform DBSCAN clustering on road centroids.

    Args:
        road_centroids: DataFrame with road median coordinates
        lat_col: Latitude column
        lon_col: Longitude column

    Returns:
        Tuple[pd.DataFrame, float]: Clustered DataFrame and noise ratio
    """
    logger.info("Performing DBSCAN clustering...")

    # Convert to radians for haversine distance
    coords = np.radians(road_centroids[[lat_col, lon_col]].values)

    # Find optimal epsilon using k-distance plot
    eps = _find_optimal_eps(coords)

    # Perform DBSCAN
    dbscan = DBSCAN(
        eps=eps,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric=DBSCAN_METRIC
    )

    labels = dbscan.fit_predict(coords)

    road_centroids['market_cluster'] = labels
    road_centroids['is_noise'] = (labels == -1)
    road_centroids['clustering_method'] = 'dbscan'

    # Calculate noise ratio
    noise_count = (labels == -1).sum()
    noise_ratio = noise_count / len(labels)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    logger.info(
        f"DBSCAN results: {n_clusters} clusters, "
        f"{noise_count} noise points ({noise_ratio:.1%})"
    )

    return road_centroids, noise_ratio


def _cluster_with_kmeans(
    road_centroids: pd.DataFrame,
    lat_col: str,
    lon_col: str
) -> pd.DataFrame:
    """
    Perform K-Means clustering on road centroids.

    Args:
        road_centroids: DataFrame with road median coordinates
        lat_col: Latitude column
        lon_col: Longitude column

    Returns:
        pd.DataFrame: Clustered DataFrame
    """
    logger.info(f"Performing K-Means clustering with k={KMEANS_N_CLUSTERS}...")

    coords = road_centroids[[lat_col, lon_col]].values

    # Perform K-Means
    kmeans = KMeans(
        n_clusters=KMEANS_N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10
    )

    labels = kmeans.fit_predict(coords)

    road_centroids['market_cluster'] = labels
    road_centroids['is_noise'] = False  # K-Means doesn't produce noise
    road_centroids['clustering_method'] = 'kmeans'

    logger.info(f"K-Means results: {KMEANS_N_CLUSTERS} clusters")

    return road_centroids


def _find_optimal_eps(coords: np.ndarray) -> float:
    """
    Find optimal epsilon for DBSCAN using k-distance plot.

    Uses the 90-95th percentile of k-nearest neighbor distances as epsilon.

    Args:
        coords: Array of coordinates (in radians for haversine)

    Returns:
        float: Optimal epsilon value
    """
    from sklearn.neighbors import NearestNeighbors

    logger.info("Finding optimal epsilon for DBSCAN...")

    # Calculate k-nearest neighbor distances
    k = DBSCAN_MIN_SAMPLES
    nbrs = NearestNeighbors(n_neighbors=k, metric=DBSCAN_METRIC)
    nbrs.fit(coords)

    distances, indices = nbrs.kneighbors(coords)

    # Get k-th nearest neighbor distance for each point
    k_distances = distances[:, -1]

    # Use percentile of k-distances as epsilon
    eps_min = np.percentile(k_distances, DBSCAN_EPS_PERCENTILE_MIN * 100)
    eps_max = np.percentile(k_distances, DBSCAN_EPS_PERCENTILE_MAX * 100)
    eps = (eps_min + eps_max) / 2

    logger.info(f"Optimal epsilon: {eps:.6f} (range: {eps_min:.6f} - {eps_max:.6f})")

    return eps


def assign_cluster_ids(
    road_centroids: pd.DataFrame,
    district_col: str = 'district'
) -> pd.DataFrame:
    """
    Assign human-readable cluster IDs with district prefixes.

    Examples: "KL_017" (Kuala Lumpur cluster 17), "PE_023" (Petaling cluster 23)

    Args:
        road_centroids: DataFrame with cluster assignments
        district_col: District column name

    Returns:
        pd.DataFrame: DataFrame with market_cluster_id column added
    """
    logger.info("Assigning cluster IDs with district prefixes...")

    road_centroids = road_centroids.copy()

    # Create cluster IDs within each district
    cluster_ids = []

    for idx, row in road_centroids.iterrows():
        district = row[district_col].lower()
        cluster_num = row['market_cluster']

        # Get district prefix
        prefix = CLUSTER_PREFIXES.get(district, 'XX')

        # Handle noise points
        if row.get('is_noise', False):
            cluster_id = 'NOISE'
        else:
            cluster_id = f"{prefix}_{cluster_num:03d}"

        cluster_ids.append(cluster_id)

    road_centroids['market_cluster_id'] = cluster_ids

    n_unique = len(set(cluster_ids))
    logger.info(f"Created {n_unique} unique market cluster IDs")

    return road_centroids


def get_cluster_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for each market cluster.

    Args:
        df: DataFrame with cluster assignments and prices

    Returns:
        pd.DataFrame: Cluster statistics
    """
    logger.info("Calculating cluster statistics...")

    stats = df.groupby('market_cluster_id').agg({
        'price_m2': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'property_m2': 'mean',
        'district': 'first'
    }).reset_index()

    stats.columns = [
        'market_cluster_id',
        'property_count',
        'avg_price_m2',
        'median_price_m2',
        'std_price_m2',
        'min_price_m2',
        'max_price_m2',
        'avg_property_size',
        'district'
    ]

    stats = stats.sort_values('property_count', ascending=False)

    logger.info(f"Generated statistics for {len(stats)} clusters")

    return stats
