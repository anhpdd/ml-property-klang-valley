"""
OpenStreetMap API helpers.

Utilities for querying OSM data with retry logic and rate limiting.
"""

import logging
import time
from typing import Optional

from ..config import (
    OSM_API_TIMEOUT,
    OSM_API_RATE_LIMIT,
    OSM_MAX_RETRIES,
    OSM_RETRY_BACKOFF
)

logger = logging.getLogger(__name__)


class OSMQueryError(Exception):
    """Raised when OSM query fails after all retries."""
    pass


# Transient errors that should be retried
TRANSIENT_ERROR_TYPES = (
    ConnectionError,
    TimeoutError,
    OSError,  # Network errors
)

# Error messages indicating rate limiting
RATE_LIMIT_MESSAGES = [
    'too many requests',
    'rate limit',
    '429',
    'quota exceeded',
]


def _is_transient_error(error: Exception) -> bool:
    """
    Check if an error is transient and should be retried.

    Args:
        error: The exception that occurred

    Returns:
        bool: True if error is transient
    """
    # Check error type
    if isinstance(error, TRANSIENT_ERROR_TYPES):
        return True

    # Check error message for rate limiting
    error_msg = str(error).lower()
    for msg in RATE_LIMIT_MESSAGES:
        if msg in error_msg:
            return True

    return False


def query_osm_with_retry(
    query_func,
    *args,
    max_retries: int = OSM_MAX_RETRIES,
    backoff: float = OSM_RETRY_BACKOFF,
    **kwargs
):
    """
    Execute OSM query with exponential backoff retry logic.

    Args:
        query_func: Function to execute (e.g., ox.geocode, ox.features_from_place)
        *args: Positional arguments for query_func
        max_retries: Maximum number of retry attempts
        backoff: Exponential backoff multiplier
        **kwargs: Keyword arguments for query_func

    Returns:
        Query results

    Raises:
        OSMQueryError: If all retries fail or non-transient error occurs
    """
    last_error = None

    for attempt in range(max_retries):
        # Rate limiting BEFORE request to prevent hitting limits
        if attempt > 0:
            wait_time = backoff ** attempt
            logger.info(f"Waiting {wait_time:.1f}s before retry...")
            time.sleep(wait_time)
        else:
            # Still apply rate limit on first request
            time.sleep(OSM_API_RATE_LIMIT)

        try:
            logger.debug(f"OSM query attempt {attempt + 1}/{max_retries}")
            result = query_func(*args, **kwargs)
            return result

        except KeyboardInterrupt:
            # Don't catch user interrupts
            raise

        except SystemExit:
            # Don't catch system exit
            raise

        except Exception as e:
            last_error = e

            # Check if error is transient
            if not _is_transient_error(e):
                logger.error(f"Non-transient OSM error (not retrying): {type(e).__name__}: {e}")
                raise OSMQueryError(f"OSM query failed with non-transient error: {e}") from e

            if attempt < max_retries - 1:
                logger.warning(
                    f"OSM query failed (attempt {attempt + 1}/{max_retries}): "
                    f"{type(e).__name__}: {e}"
                )
            else:
                logger.error(f"OSM query failed after {max_retries} attempts: {e}")

    raise OSMQueryError(
        f"OSM query failed after {max_retries} attempts. Last error: {last_error}"
    ) from last_error


def check_osm_rate_limit() -> None:
    """
    Check OSM API rate limit by enforcing minimum wait time.

    Call this function between OSM API requests to avoid hitting rate limits.
    """
    time.sleep(OSM_API_RATE_LIMIT)


def geocode_with_osm(address: str, retries: int = OSM_MAX_RETRIES) -> Optional[dict]:
    """
    Geocode an address using OSM Nominatim.

    Args:
        address: Address string to geocode
        retries: Number of retry attempts

    Returns:
        Optional[dict]: Geocoding result or None if failed

    Note:
        This is a placeholder. For full implementation, use osmnx.geocode()
        See notebooks/0_Geocode_Names_to_Way_ID.ipynb for reference.
    """
    logger.warning("geocode_with_osm not fully implemented - placeholder only")

    # TODO: Implement using osmnx
    # Example:
    # try:
    #     import osmnx as ox
    #     result = query_osm_with_retry(ox.geocode, address)
    #     return result
    # except Exception as e:
    #     logger.error(f"Geocoding failed for {address}: {e}")
    #     return None

    return None


def get_amenities_from_place(
    place_name: str,
    tags: dict,
    retries: int = OSM_MAX_RETRIES
):
    """
    Query OSM for amenities in a place.

    Args:
        place_name: Name of place to query
        tags: OSM tags dictionary (e.g., {'amenity': 'school'})
        retries: Number of retry attempts

    Returns:
        GeoDataFrame with amenities

    Note:
        This is a placeholder. For full implementation, use osmnx.features_from_place()
        See notebooks/2_1_Amenity_OSM_search.ipynb for reference.
    """
    logger.warning("get_amenities_from_place not fully implemented - placeholder only")

    # TODO: Implement using osmnx
    # Example:
    # import osmnx as ox
    # amenities = query_osm_with_retry(
    #     ox.features_from_place,
    #     place_name,
    #     tags=tags
    # )
    # return amenities

    return None


def download_road_network(
    place_name: str,
    network_type: str = 'walk',
    retries: int = OSM_MAX_RETRIES
):
    """
    Download road network graph from OSM.

    Args:
        place_name: Name of place to query
        network_type: Type of network ('walk', 'drive', 'bike', 'all')
        retries: Number of retry attempts

    Returns:
        NetworkX graph

    Note:
        This is a placeholder. For full implementation, use osmnx.graph_from_place()
        See notebooks/2_1_Amenity_OSM_search.ipynb for reference.
    """
    logger.warning("download_road_network not fully implemented - placeholder only")

    # TODO: Implement using osmnx
    # Example:
    # import osmnx as ox
    # G = query_osm_with_retry(
    #     ox.graph_from_place,
    #     place_name,
    #     network_type=network_type
    # )
    # return G

    return None
