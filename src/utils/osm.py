"""
OpenStreetMap API helpers.

Utilities for querying OSM data with retry logic and rate limiting.
"""

import logging
import time
from typing import Optional, Dict, List, Any
import requests
import xml.etree.ElementTree as ET
import overpy

from ..config import (
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
                logger.error(
                    f"Non-transient OSM error (not retrying): {type(e).__name__}: {e}")
                raise OSMQueryError(
                    f"OSM query failed with non-transient error: {e}") from e

            if attempt < max_retries - 1:
                logger.warning(
                    f"OSM query failed (attempt {attempt + 1}/{max_retries}): "
                    f"{type(e).__name__}: {e}"
                )
            else:
                logger.error(
                    f"OSM query failed after {max_retries} attempts: {e}")

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


def build_overpass_query(area_ids: Dict[str, str], categories: Dict[str, Dict[str, str]]) -> str:
    """
    Builds a correctly formatted Overpass QL query for multiple areas and categories.

    Args:
        area_ids: Dictionary mapping area names to their OSM relation IDs.
        categories: Dictionary mapping category names to their OSM tag filters.

    Returns:
        str: The Overpass QL query string.
    """
    query_parts = []

    for area_name, rel_id in area_ids.items():
        area_id_for_query = int(rel_id) + 3600000000

        for category_name, tag_dict in categories.items():
            tag_filters = ''.join(
                [f'["{k}"="{v}"]' for k, v in tag_dict.items()])
            query_parts.append(
                f'  nwr{tag_filters}(area:{area_id_for_query});')

    query_body = '\n'.join(query_parts)

    full_query = f"""
[out:json][timeout:240];
(
{query_body}
);
(._;>>;);
out center;
"""
    return full_query


def parse_overpy_element(element, categories: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """
    Parses a single overpy element into a structured dictionary.

    Args:
        element: The overpy element (Node, Way, or Relation).
        categories: POI category definitions for matching.

    Returns:
        dict: Parsed POI data.
    """
    category = "Unknown"

    for cat_name, tag_dict in categories.items():
        if all(element.tags.get(k) == v for k, v in tag_dict.items()):
            category = cat_name
            break

    geometry = None
    if isinstance(element, overpy.Node):
        geometry = (float(element.lat), float(element.lon))
    elif isinstance(element, overpy.Way):
        if element.nodes:
            geometry = [(float(node.lat), float(node.lon))
                        for node in element.nodes]
    elif isinstance(element, overpy.Relation):
        if hasattr(element, 'center_lat') and element.center_lat is not None:
            geometry = (float(element.center_lat), float(element.center_lon))

    return {
        "osm_id": element.id,
        "osm_type": element.__class__.__name__.lower(),
        "name": element.tags.get("name", "N/A"),
        "category": category,
        "tags": dict(element.tags),
        "geometry_coords": geometry
    }


def get_amenities_from_osm(area_ids: Dict[str, str], categories: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Query Overpass API for amenities in specified areas.

    Args:
        area_ids: Dictionary of area names and OSM IDs.
        categories: Dictionary of POI categories and tags.

    Returns:
        list: List of parsed POI dictionaries.
    """
    query = build_overpass_query(area_ids, categories)
    api = overpy.Overpass()

    logger.info(
        f"Querying Overpass API for {len(categories)} types in {len(area_ids)} areas")

    try:
        result = query_osm_with_retry(api.query, query)

        all_pois = []
        for elements in [result.nodes, result.ways, result.relations]:
            for element in elements:
                poi = parse_overpy_element(element, categories)
                if poi['category'] != 'Unknown' and poi['geometry_coords'] is not None:
                    all_pois.append(poi)

        logger.info(f"Successfully parsed {len(all_pois)} POIs")
        return all_pois

    except Exception as e:
        logger.error(f"Failed to fetch amenities: {e}")
        return []


def get_node_coordinates(node_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetches the coordinates and metadata of a node from OSM API.

    Args:
        node_id: The OSM node ID.

    Returns:
        dict: Node metadata including lat, lon, name, and ref.
    """
    url = f"https://api.openstreetmap.org/api/0.6/node/{node_id}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        node = root.find('node')
        if node is not None:
            lat = node.get('lat')
            lon = node.get('lon')
            name = None
            ref = None
            for tag in node.findall('tag'):
                k = tag.get('k')
                v = tag.get('v')
                if k == 'name':
                    name = v
                elif k == 'ref':
                    ref = v
            return {
                'lat': float(lat) if lat else None,
                'lon': float(lon) if lon else None,
                'name': name,
                'ref': ref
            }
        return None

    except Exception as e:
        logger.warning(f"Error processing node {node_id}: {e}")
        return None


def get_train_route_data(train_id: str) -> List[Dict[str, Any]]:
    """
    Fetches stop data for a given train line relation ID.

    Args:
        train_id: The OSM relation ID for the train route.

    Returns:
        list: List of station metadata dictionaries.
    """
    url = f'https://www.openstreetmap.org/api/0.6/relation/{train_id}'
    stations = []

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        for relation in root.findall('relation'):
            # Basic route info
            route_name = None
            route_ref = None
            for tag in relation.findall('tag'):
                if tag.get('k') == 'name':
                    route_name = tag.get('v')
                if tag.get('k') == 'ref':
                    route_ref = tag.get('v')

            for member in relation.findall('member'):
                if member.get('type') == 'node':
                    node_id = member.get('ref')
                    node_data = get_node_coordinates(node_id)
                    if node_data:
                        node_data['node_id'] = node_id
                        node_data['route_name'] = route_name
                        node_data['route_ref'] = route_ref
                        stations.append(node_data)

        return stations

    except Exception as e:
        logger.error(f"Error fetching train data for {train_id}: {e}")
        return []


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
    logger.warning(
        "download_road_network not fully implemented - placeholder only")

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
