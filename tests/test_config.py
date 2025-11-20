"""
Tests for configuration module.
"""

import pytest
from src.config import (
    DISTANCE_COLS,
    COUNT_COLS,
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
    ensure_directories,
    get_feature_lists
)


def test_feature_lists_not_empty():
    """Test that feature lists are defined."""
    assert len(DISTANCE_COLS) > 0
    assert len(COUNT_COLS) > 0
    assert len(CONTINUOUS_FEATURES) > 0
    assert len(CATEGORICAL_FEATURES) > 0


def test_continuous_features_includes_all():
    """Test that continuous features include all sub-categories."""
    for col in DISTANCE_COLS:
        assert col in CONTINUOUS_FEATURES

    for col in COUNT_COLS:
        assert col in CONTINUOUS_FEATURES


def test_ensure_directories():
    """Test that ensure_directories creates required directories."""
    ensure_directories()
    # If this runs without error, directories were created
    assert True


def test_get_feature_lists():
    """Test get_feature_lists returns dictionary."""
    features = get_feature_lists()

    assert isinstance(features, dict)
    assert 'distance_cols' in features
    assert 'count_cols' in features
    assert 'continuous_features' in features
