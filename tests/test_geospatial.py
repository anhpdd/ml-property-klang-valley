"""
Tests for geospatial feature extraction.
"""

import pytest
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from src.features.geospatial import count_amenities_within_radius, calculate_haversine_distance

def test_calculate_haversine_distance():
    """Test haversine distance calculation."""
    # Distance between two points in KL (approx 1.1km)
    lat1, lon1 = 3.1390, 101.6869
    lat2, lon2 = 3.1490, 101.6869

    dist = calculate_haversine_distance(lat1, lon1, lat2, lon2)
    assert dist == pytest.approx(1.11, abs=0.01)

def test_count_amenities_within_radius():
    """Test counting amenities within radius."""
    # Create sample properties
    properties_df = pd.DataFrame({
        'id': [1, 2],
        'latitude': [3.1, 3.2],
        'longitude': [101.6, 101.7]
    })
    properties_gdf = gpd.GeoDataFrame(
        properties_df,
        geometry=gpd.points_from_xy(properties_df.longitude, properties_df.latitude),
        crs="EPSG:4326"
    )

    # Create sample amenities
    amenities_df = pd.DataFrame({
        'feature_type': ['school', 'school', 'mall'],
        'latitude': [3.101, 3.102, 3.201],
        'longitude': [101.601, 101.602, 101.701]
    })
    amenities_gdf = gpd.GeoDataFrame(
        amenities_df,
        geometry=gpd.points_from_xy(amenities_df.longitude, amenities_df.latitude),
        crs="EPSG:4326"
    )

    # Count amenities within 1km
    result_gdf = count_amenities_within_radius(properties_gdf, amenities_gdf, radius_km=1.0)

    assert 'school_count' in result_gdf.columns
    assert 'mall_count' in result_gdf.columns

    # Property 1 (3.1, 101.6) should have 2 schools and 0 malls nearby
    assert result_gdf.loc[0, 'school_count'] == 2
    assert result_gdf.loc[0, 'mall_count'] == 0

    # Property 2 (3.2, 101.7) should have 0 schools and 1 mall nearby
    assert result_gdf.loc[1, 'school_count'] == 0
    assert result_gdf.loc[1, 'mall_count'] == 1
