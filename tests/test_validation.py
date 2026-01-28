"""
Tests for spatial validation logic.
"""

import pytest
from shapely.geometry import Polygon, LineString
from src.data.validation import stitch_relation_polygons

def test_stitch_relation_polygons():
    """Test stitching way segments into polygons."""
    # Create simple outer square
    outer_segments = [
        [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    ]
    inner_segments = []

    polygons = stitch_relation_polygons(outer_segments, inner_segments)

    assert len(polygons) == 1
    assert isinstance(polygons[0], Polygon)
    assert polygons[0].area == 1.0

def test_stitch_relation_polygons_with_hole():
    """Test stitching polygons with holes."""
    outer_segments = [
        [(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)]
    ]
    inner_segments = [
        [(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)]
    ]

    polygons = stitch_relation_polygons(outer_segments, inner_segments)

    assert len(polygons) == 1
    # Area: 3*3 - 1*1 = 8
    assert polygons[0].area == 8.0
    assert len(polygons[0].interiors) == 1
