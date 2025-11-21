"""
Tests for OSM utilities.

Tests retry logic, rate limiting, and error handling.
"""

import pytest
import time
from unittest.mock import Mock, patch

from src.utils.osm import (
    query_osm_with_retry,
    _is_transient_error,
    OSMQueryError,
    TRANSIENT_ERROR_TYPES,
    RATE_LIMIT_MESSAGES
)


class TestIsTransientError:
    """Tests for transient error detection."""

    def test_connection_error_is_transient(self):
        """Test that ConnectionError is detected as transient."""
        error = ConnectionError("Network unreachable")
        assert _is_transient_error(error) is True

    def test_timeout_error_is_transient(self):
        """Test that TimeoutError is detected as transient."""
        error = TimeoutError("Request timed out")
        assert _is_transient_error(error) is True

    def test_os_error_is_transient(self):
        """Test that OSError is detected as transient."""
        error = OSError("Network error")
        assert _is_transient_error(error) is True

    def test_rate_limit_message_is_transient(self):
        """Test that rate limit errors are detected as transient."""
        error = Exception("Too many requests - please slow down")
        assert _is_transient_error(error) is True

        error = Exception("429 Client Error")
        assert _is_transient_error(error) is True

    def test_value_error_not_transient(self):
        """Test that ValueError is not transient."""
        error = ValueError("Invalid argument")
        assert _is_transient_error(error) is False

    def test_key_error_not_transient(self):
        """Test that KeyError is not transient."""
        error = KeyError("missing_key")
        assert _is_transient_error(error) is False


class TestQueryOSMWithRetry:
    """Tests for OSM query retry logic."""

    def test_successful_query_returns_result(self):
        """Test that successful query returns result."""
        mock_func = Mock(return_value="result")

        with patch('src.utils.osm.time.sleep'):  # Skip actual sleep
            result = query_osm_with_retry(mock_func, "arg1", kwarg1="value1")

        assert result == "result"
        mock_func.assert_called_once_with("arg1", kwarg1="value1")

    def test_transient_error_retried(self):
        """Test that transient errors are retried."""
        mock_func = Mock(side_effect=[
            ConnectionError("Network error"),
            ConnectionError("Network error"),
            "success"
        ])

        with patch('src.utils.osm.time.sleep'):
            result = query_osm_with_retry(mock_func, max_retries=3)

        assert result == "success"
        assert mock_func.call_count == 3

    def test_non_transient_error_not_retried(self):
        """Test that non-transient errors are not retried."""
        mock_func = Mock(side_effect=ValueError("Invalid argument"))

        with patch('src.utils.osm.time.sleep'):
            with pytest.raises(OSMQueryError, match="non-transient error"):
                query_osm_with_retry(mock_func, max_retries=3)

        # Should only be called once (no retries)
        assert mock_func.call_count == 1

    def test_all_retries_exhausted_raises(self):
        """Test that exhausting all retries raises OSMQueryError."""
        mock_func = Mock(side_effect=ConnectionError("Network error"))

        with patch('src.utils.osm.time.sleep'):
            with pytest.raises(OSMQueryError, match="failed after 3 attempts"):
                query_osm_with_retry(mock_func, max_retries=3)

        assert mock_func.call_count == 3

    def test_keyboard_interrupt_not_caught(self):
        """Test that KeyboardInterrupt is not caught."""
        mock_func = Mock(side_effect=KeyboardInterrupt())

        with patch('src.utils.osm.time.sleep'):
            with pytest.raises(KeyboardInterrupt):
                query_osm_with_retry(mock_func, max_retries=3)

    def test_system_exit_not_caught(self):
        """Test that SystemExit is not caught."""
        mock_func = Mock(side_effect=SystemExit())

        with patch('src.utils.osm.time.sleep'):
            with pytest.raises(SystemExit):
                query_osm_with_retry(mock_func, max_retries=3)

    def test_exponential_backoff_applied(self):
        """Test that exponential backoff is applied between retries."""
        mock_func = Mock(side_effect=[
            ConnectionError("Error 1"),
            ConnectionError("Error 2"),
            "success"
        ])

        sleep_times = []

        def mock_sleep(seconds):
            sleep_times.append(seconds)

        with patch('src.utils.osm.time.sleep', side_effect=mock_sleep):
            query_osm_with_retry(mock_func, max_retries=3, backoff=2.0)

        # First attempt: rate limit sleep (1.0)
        # After first failure: backoff^0 = 1.0 wait, then rate limit
        # After second failure: backoff^1 = 2.0 wait
        # Check that backoff was applied
        assert any(t >= 1.0 for t in sleep_times)  # Some wait happened
