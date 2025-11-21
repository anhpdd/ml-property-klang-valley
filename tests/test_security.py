"""
Security tests for the ML Property Valuation project.

Tests path traversal prevention, input validation, and other security measures.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import hashlib

from src.data.loaders import (
    load_raw_data,
    _validate_file_path,
    _validate_file_size,
    DataSecurityError,
    MAX_FILE_SIZE_BYTES
)
from src.config import PROJECT_ROOT, DATA_RAW_DIR

# Try to import predictor - may fail if ML deps not installed
try:
    from src.models.predictor import (
        SecurityError,
        _verify_file_hash,
        EXPECTED_FEATURE_COUNT,
        PropertyPredictor
    )
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False
    EXPECTED_FEATURE_COUNT = 279  # Default value


class TestPathTraversal:
    """Tests for path traversal prevention."""

    def test_validate_file_path_within_project(self):
        """Test that paths within project are allowed."""
        test_file = DATA_RAW_DIR / "test_valid.csv"

        # Should not raise if file would be in allowed location
        try:
            _validate_file_path(test_file, check_allowed_dirs=False)
        except DataSecurityError:
            pytest.fail("Valid path within project should not raise DataSecurityError")

    def test_validate_file_path_traversal_blocked(self):
        """Test that path traversal attempts are blocked."""
        # Try to access file outside project root
        malicious_paths = [
            Path("../../../etc/passwd"),
            Path("..\\..\\..\\Windows\\System32\\config\\SAM"),
        ]

        for malicious_path in malicious_paths:
            # Skip if path would be within project (edge case on some systems)
            try:
                malicious_path.resolve().relative_to(PROJECT_ROOT.resolve())
                continue  # Path is actually within project, skip
            except ValueError:
                pass  # Path is outside project, should be blocked

            with pytest.raises(DataSecurityError, match="Path traversal detected"):
                _validate_file_path(malicious_path)


class TestFileSizeValidation:
    """Tests for file size limits."""

    def test_validate_small_file_passes(self, tmp_path):
        """Test that small files pass validation."""
        small_file = tmp_path / "small.csv"
        small_file.write_text("a,b,c\n1,2,3\n")

        # Should not raise
        _validate_file_size(small_file)

    def test_validate_large_file_blocked(self, tmp_path):
        """Test that oversized files are blocked."""
        large_file = tmp_path / "large.csv"
        large_file.write_text("a,b,c\n1,2,3\n")

        # Validate against tiny limit
        with pytest.raises(DataSecurityError, match="exceeds limit"):
            _validate_file_size(large_file, max_size_bytes=1)

    def test_load_raw_data_respects_size_limit(self, tmp_path):
        """Test that load_raw_data enforces size limits."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("col1,col2\n1,2\n3,4\n")

        # Should work with high limit
        df = load_raw_data(test_file, validate_path=False, max_size_mb=100)
        assert len(df) == 2

        # Should fail with tiny limit
        with pytest.raises(DataSecurityError):
            load_raw_data(test_file, validate_path=False, max_size_mb=0.000001)


@pytest.mark.skipif(not PREDICTOR_AVAILABLE, reason="Predictor not available (ML deps missing)")
class TestFeatureValidation:
    """Tests for feature validation in predictor."""

    def test_feature_count_mismatch_raises(self):
        """Test that wrong feature count raises error."""
        predictor = PropertyPredictor()

        # Create array with wrong number of features
        wrong_features = np.random.rand(10, 100)  # 100 instead of 279

        with pytest.raises(ValueError, match="Feature count mismatch"):
            predictor._validate_features(wrong_features)

    def test_correct_feature_count_passes(self):
        """Test that correct feature count passes validation."""
        predictor = PropertyPredictor()

        # Create array with correct number of features
        correct_features = np.random.rand(10, EXPECTED_FEATURE_COUNT)

        # Should not raise
        predictor._validate_features(correct_features)

    def test_nan_values_rejected(self):
        """Test that NaN values are rejected."""
        predictor = PropertyPredictor()

        # Create array with NaN
        features_with_nan = np.random.rand(10, EXPECTED_FEATURE_COUNT)
        features_with_nan[5, 100] = np.nan

        with pytest.raises(ValueError, match="NaN values"):
            predictor._validate_features(features_with_nan)

    def test_inf_values_rejected(self):
        """Test that infinite values are rejected."""
        predictor = PropertyPredictor()

        # Create array with infinity
        features_with_inf = np.random.rand(10, EXPECTED_FEATURE_COUNT)
        features_with_inf[3, 50] = np.inf

        with pytest.raises(ValueError, match="infinite values"):
            predictor._validate_features(features_with_inf)


@pytest.mark.skipif(not PREDICTOR_AVAILABLE, reason="Predictor not available (ML deps missing)")
class TestHashVerification:
    """Tests for model file hash verification."""

    def test_hash_verification_skipped_when_none(self, tmp_path):
        """Test that hash verification is skipped when expected_hash is None."""
        test_file = tmp_path / "model.pkl"
        test_file.write_bytes(b"test model content")

        # Should return True when no hash provided
        result = _verify_file_hash(test_file, expected_hash=None)
        assert result is True

    def test_hash_verification_passes_correct_hash(self, tmp_path):
        """Test that correct hash passes verification."""
        test_file = tmp_path / "model.pkl"
        content = b"test model content"
        test_file.write_bytes(content)

        # Calculate correct hash
        expected_hash = hashlib.sha256(content).hexdigest()

        result = _verify_file_hash(test_file, expected_hash=expected_hash)
        assert result is True

    def test_hash_verification_fails_wrong_hash(self, tmp_path):
        """Test that wrong hash fails verification."""
        test_file = tmp_path / "model.pkl"
        test_file.write_bytes(b"test model content")

        wrong_hash = "0" * 64  # Wrong hash

        with pytest.raises(SecurityError, match="integrity check failed"):
            _verify_file_hash(test_file, expected_hash=wrong_hash)


class TestInputValidation:
    """Tests for general input validation."""

    def test_empty_dataframe_handling(self, tmp_path):
        """Test handling of empty CSV files."""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("col1,col2\n")  # Headers only

        df = load_raw_data(empty_file, validate_path=False)
        assert len(df) == 0

    def test_invalid_file_type_rejected(self, tmp_path):
        """Test that unsupported file types are rejected."""
        bad_file = tmp_path / "data.xyz"
        bad_file.write_text("some content")

        with pytest.raises(ValueError, match="Cannot auto-detect"):
            load_raw_data(bad_file, validate_path=False)

    def test_nonexistent_file_raises(self):
        """Test that nonexistent files raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_raw_data("nonexistent_file_12345.csv", validate_path=False)
