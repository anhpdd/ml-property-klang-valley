"""
Tests for spatial clustering and market segmentation.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.clustering import assign_cluster_ids

def test_assign_cluster_ids():
    """Test assigning human-readable cluster IDs."""
    road_centroids = pd.DataFrame({
        'district': ['GOMBAK', 'PETALING', 'KUALA LUMPUR'],
        'market_cluster': [1, 23, 17],
        'is_noise': [False, False, False]
    })

    result = assign_cluster_ids(road_centroids)

    assert 'market_cluster_id' in result.columns
    assert result.loc[0, 'market_cluster_id'] == 'GO_001'
    assert result.loc[1, 'market_cluster_id'] == 'PE_023'
    assert result.loc[2, 'market_cluster_id'] == 'KL_017'

def test_assign_cluster_ids_noise():
    """Test assigning NOISE ID to noise points."""
    road_centroids = pd.DataFrame({
        'district': ['GOMBAK'],
        'market_cluster': [-1],
        'is_noise': [True]
    })

    result = assign_cluster_ids(road_centroids)

    assert result.loc[0, 'market_cluster_id'] == 'NOISE'
