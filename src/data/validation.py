import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

"""
OSM Geometry Extraction and Spatial Validation Module

Handles:
1. Fetching road geometries from OpenStreetMap API
2. Processing complex OSM relations with holes
3. Creating district boundary GeoDataFrames
4. Validating road locations against district boundaries
5. Caching for performance optimization
"""

# ===== IMPORTS =====
# Standard library
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

# Third-party
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import xml.etree.ElementTree as ET

# Geospatial
from shapely.geometry import (
    Point, LineString, Polygon, MultiPolygon,
    GeometryCollection
)
from shapely.ops import unary_union, polygonize

# Progress tracking
from tqdm.auto import tqdm

# Local
from src.config import CLEANED_DATA_DIR, SUPPORT_DATA_DIR, DISTRICT_OSM_IDS


# ===== CONFIGURATION =====
@dataclass
class OSMConfig:
    """Configuration for OpenStreetMap API interactions."""
    
    api_base_url: str = "https://api.openstreetmap.org/api/0.6"
    user_agent: str = "SelangorPropertyAnalysis/1.0 (Research Project)"
    timeout: int = 25
    cache_dir: Path = SUPPORT_DATA_DIR / "osm_cache"
    cache_ttl_days: int = 30  # Cache validity period
    
    def __post_init__(self):
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def headers(self) -> Dict[str, str]:
        """HTTP headers for API requests."""
        return {'User-Agent': self.user_agent}


# District OSM IDs (stable reference data)



# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===== CACHING UTILITIES =====
class OSMCache:
    """Simple file-based cache for OSM API responses."""
    
    def __init__(self, config: OSMConfig):
        self.config = config
        self.cache_dir = config.cache_dir
    
    def _get_cache_path(self, osm_type: str, osm_id: str) -> Path:
        """Generate cache file path for an OSM object."""
        return self.cache_dir / f"{osm_type}_{osm_id}.pkl"
    
    def get(self, osm_type: str, osm_id: str) -> Optional[ET.Element]:
        """Retrieve cached OSM data if valid."""
        cache_path = self._get_cache_path(osm_type, osm_id)
        
        if not cache_path.exists():
            return None
        
        # Check if cache is expired
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age > timedelta(days=self.config.cache_ttl_days):
            logger.debug(f"Cache expired for {osm_type}/{osm_id}")
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache for {osm_type}/{osm_id}: {e}")
            return None
    
    def set(self, osm_type: str, osm_id: str, data: ET.Element) -> None:
        """Cache OSM data."""
        cache_path = self._get_cache_path(osm_type, osm_id)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to cache {osm_type}/{osm_id}: {e}")


# ===== OSM API FUNCTIONS =====
def fetch_osm_data(
    url: str,
    config: OSMConfig,
    cache: Optional[OSMCache] = None
) -> Optional[ET.Element]:
    """
    Fetch data from OSM API with caching support.
    
    Args:
        url: Full API endpoint URL
        config: OSM configuration
        cache: Optional cache instance
    
    Returns:
        Parsed XML element or None if request failed
    """
    try:
        response = requests.get(
            url,
            timeout=config.timeout,
            headers=config.headers
        )
        response.raise_for_status()
        return ET.fromstring(response.content)
        
    except requests.exceptions.HTTPError as e:
        if e.response and e.response.status_code == 404:
            logger.debug(f"OSM object not found: {url}")
            return None
        logger.error(f"HTTP error for {url}: {e}")
        return None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {url}: {e}")
        return None
        
    except ET.ParseError as e:
        logger.error(f"XML parsing failed for {url}: {e}")
        return None


def fetch_way_geometry(
    way_id: Union[int, str],
    config: OSMConfig,
    cache: Optional[OSMCache] = None
) -> Optional[ET.Element]:
    """
    Fetch full way data from OSM API with caching.
    
    Args:
        way_id: OSM way ID
        config: OSM configuration
        cache: Optional cache instance
    
    Returns:
        XML root element or None
    """
    way_id_str = str(int(way_id))
    
    # Check cache first
    if cache:
        cached = cache.get('way', way_id_str)
        if cached is not None:
            return cached
    
    # Fetch from API
    url = f"{config.api_base_url}/way/{way_id_str}/full"
    root = fetch_osm_data(url, config)
    
    # Cache the result
    if root is not None and cache:
        cache.set('way', way_id_str, root)
    
    return root


# ===== GEOMETRY EXTRACTION FUNCTIONS =====
def extract_line_from_way(
    root: ET.Element,
    way_id: str
) -> Optional[LineString]:
    """
    Extract LineString geometry from OSM way XML.
    
    Args:
        root: XML root element from OSM API
        way_id: Target way ID
    
    Returns:
        LineString or None if extraction failed
    """
    # Find the way element
    way_elem = root.find(f".//way[@id='{way_id}']")
    if way_elem is None:
        return None
    
    # Build node coordinate cache
    node_cache = {}
    for node in root.findall(".//node"):
        try:
            node_id = node.get('id')
            lat = float(node.get('lat'))
            lon = float(node.get('lon'))
            node_cache[node_id] = (lon, lat)
        except (TypeError, ValueError):
            continue
    
    # Extract coordinates
    coords = []
    for nd_ref in way_elem.findall('nd'):
        node_ref = nd_ref.get('ref')
        if node_ref in node_cache:
            coords.append(node_cache[node_ref])
    
    # Create LineString if valid
    if len(coords) >= 2:
        return LineString(coords)
    
    return None


def extract_geometries_for_ways(
    df: pd.DataFrame,
    id_column: str,
    name_column: str,
    config: OSMConfig
) -> pd.DataFrame:
    """
    Extract road geometries for all way IDs in DataFrame.
    
    Args:
        df: DataFrame with way IDs
        id_column: Column containing OSM way IDs
        name_column: Column containing road names (for logging)
        config: OSM configuration
    
    Returns:
        DataFrame with added 'geometry' column
    """
    logger.info(f"📍 Extracting geometry for {len(df):,} way IDs...")
    
    cache = OSMCache(config)
    way_id_to_geometry = {}
    
    unique_way_ids = df[id_column].dropna().unique()
    
    for way_id in tqdm(unique_way_ids, desc="Fetching road geometries"):
        way_id_str = str(int(way_id))
        
        # Fetch OSM data (with caching)
        root = fetch_way_geometry(way_id, config, cache)
        if root is None:
            continue
        
        # Extract geometry
        line_geom = extract_line_from_way(root, way_id_str)
        if line_geom:
            way_id_to_geometry[way_id] = line_geom
    
    # Map geometries back to DataFrame
    result_df = df.copy()
    result_df['geometry'] = result_df[id_column].map(way_id_to_geometry)
    
    # Summary statistics
    success_count = result_df['geometry'].notna().sum()
    success_rate = success_count / len(result_df) * 100
    
    logger.info(f"✅ Extracted {success_count:,}/{len(df):,} geometries ({success_rate:.1f}%)")
    
    return result_df


# ===== COMPLEX RELATION PROCESSING =====
def stitch_relation_polygons(
    outer_segments: List[List[Tuple]],
    inner_segments: List[List[Tuple]]
) -> List[Polygon]:
    """
    Stitch outer and inner way segments into valid Polygons with holes.
    
    Uses Shapely's unary_union and polygonize for robust stitching.
    
    Args:
        outer_segments: List of coordinate lists for outer boundaries
        inner_segments: List of coordinate lists for inner boundaries (holes)
    
    Returns:
        List of Shapely Polygon objects with holes
    """
    if not outer_segments:
        return []
    
    # Convert to LineStrings
    outer_lines = [LineString(seg) for seg in outer_segments if len(seg) >= 2]
    inner_lines = [LineString(seg) for seg in inner_segments if len(seg) >= 2]
    
    if not outer_lines:
        return []
    
    # Merge connected lines
    merged_outer = unary_union(outer_lines)
    merged_inner = unary_union(inner_lines) if inner_lines else None
    
    # Create polygons from closed rings
    outer_polygons = list(polygonize(merged_outer))
    inner_polygons = list(polygonize(merged_inner)) if merged_inner else []
    
    # Assign holes to their containing outer polygons
    final_polygons = []
    remaining_inners = list(inner_polygons)
    
    for outer_poly in outer_polygons:
        holes = []
        unassigned_inners = []
        
        for inner_poly in remaining_inners:
            if outer_poly.contains(inner_poly):
                holes.append(inner_poly.exterior.coords)
            else:
                unassigned_inners.append(inner_poly)
        
        remaining_inners = unassigned_inners
        
        # Create polygon with holes
        final_polygons.append(Polygon(outer_poly.exterior.coords, holes))
    
    return final_polygons


def fetch_relation_geometry(
    relation_id: str,
    config: OSMConfig,
    cache: Optional[OSMCache] = None
) -> Optional[Dict]:
    """
    Fetch and process OSM relation into polygon geometry.
    
    Args:
        relation_id: OSM relation ID
        config: OSM configuration
        cache: Optional cache instance
    
    Returns:
        Dictionary with geometry data or None
    """
    # Check cache
    if cache:
        cached = cache.get('relation', relation_id)
        if cached is not None:
            root = cached
        else:
            url = f"{config.api_base_url}/relation/{relation_id}/full"
            root = fetch_osm_data(url, config)
            if root:
                cache.set('relation', relation_id, root)
    else:
        url = f"{config.api_base_url}/relation/{relation_id}/full"
        root = fetch_osm_data(url, config)
    
    if root is None:
        return None
    
    # Find relation element
    relation_elem = root.find(f".//relation[@id='{relation_id}']")
    if relation_elem is None:
        return None
    
    # Extract tags
    tags = {tag.get('k'): tag.get('v') 
            for tag in relation_elem.findall('tag') 
            if tag.get('k')}
    
    # Build node coordinate cache
    node_cache = {}
    for node in root.findall('.//node'):
        try:
            node_id = node.get('id')
            lat = float(node.get('lat'))
            lon = float(node.get('lon'))
            node_cache[node_id] = (lon, lat)
        except (TypeError, ValueError):
            continue
    
    # Separate outer and inner ways
    outer_segments = []
    inner_segments = []
    
    for member in relation_elem.findall("member[@type='way']"):
        way_elem = root.find(f".//way[@id='{member.get('ref')}']")
        if way_elem is None:
            continue
        
        coords = [
            node_cache[nd.get('ref')] 
            for nd in way_elem.findall('nd') 
            if nd.get('ref') in node_cache
        ]
        
        if coords:
            role = member.get('role', 'outer')
            if role == 'outer':
                outer_segments.append(coords)
            elif role == 'inner':
                inner_segments.append(coords)
    
    # Stitch polygons
    polygons = stitch_relation_polygons(outer_segments, inner_segments)
    
    if not polygons:
        logger.warning(f"Could not form valid polygons for relation {relation_id}")
        return None
    
    # Format polygon data
    all_polygons_coords = []
    for poly in polygons:
        exterior_coords = list(poly.exterior.coords)
        if not poly.exterior.is_ccw:
            exterior_coords.reverse()
        
        poly_data = [exterior_coords]
        
        for interior in poly.interiors:
            interior_coords = list(interior.coords)
            if interior.is_ccw:
                interior_coords.reverse()
            poly_data.append(interior_coords)
        
        all_polygons_coords.append(poly_data)
    
    logger.info(f"✓ Processed relation {relation_id} with {len(polygons)} polygon(s)")
    
    return {
        'id': relation_id,
        'type': 'relation',
        'name': tags.get('name'),
        'tags': tags,
        'all_polygons_coordinates': all_polygons_coords
    }


# ===== GEODATAFRAME CREATION =====
def create_geometry_from_coords(row: pd.Series) -> Optional[Union[Point, LineString, Polygon, MultiPolygon]]:
    """
    Create Shapely geometry from coordinate data.
    
    Args:
        row: DataFrame row with 'type' and 'all_polygons_coordinates'
    
    Returns:
        Shapely geometry object or None
    """
    coords_list = row.get('all_polygons_coordinates', [])
    obj_type = row.get('type')
    obj_id = row.get('id', 'unknown')
    
    if not coords_list:
        return None
    
    try:
        if obj_type == 'node':
            if coords_list and len(coords_list[0]) == 1:
                return Point(coords_list[0][0])
        
        elif obj_type == 'way_polygon':
            if coords_list and coords_list[0] and len(coords_list[0]) >= 4:
                coords = coords_list[0]
                if coords[0] != coords[-1]:
                    coords = coords + [coords[0]]
                return Polygon(coords)
        
        elif obj_type == 'way_line':
            if coords_list and coords_list[0] and len(coords_list[0]) >= 2:
                return LineString(coords_list[0])
        
        elif obj_type == 'relation':
            polygons = []
            for poly_data in coords_list:
                if not poly_data:
                    continue
                
                exterior = poly_data[0]
                holes = poly_data[1:] if len(poly_data) > 1 else []
                
                poly = Polygon(exterior, holes)
                
                if not poly.is_valid:
                    poly = poly.buffer(0)  # Fix invalid geometry
                
                if poly.is_valid:
                    if isinstance(poly, MultiPolygon):
                        polygons.extend(poly.geoms)
                    else:
                        polygons.append(poly)
            
            if polygons:
                return MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]
    
    except Exception as e:
        logger.warning(f"Error creating geometry for {obj_type} {obj_id}: {e}")
    
    return None


def create_district_geodataframe(
    district_dict: Dict[str, str],
    config: OSMConfig
) -> gpd.GeoDataFrame:
    """
    Create GeoDataFrame of district boundaries from OSM relations.
    
    Args:
        district_dict: Mapping of district names to OSM relation IDs
        config: OSM configuration
    
    Returns:
        GeoDataFrame with district boundaries
    """
    logger.info(f"📍 Fetching {len(district_dict)} district boundaries from OSM...")
    
    cache = OSMCache(config)
    district_data = []
    
    for district_name, osm_id in tqdm(district_dict.items(), desc="Fetching districts"):
        result = fetch_relation_geometry(osm_id, config, cache)
        if result:
            result['district'] = district_name
            district_data.append(result)
        else:
            logger.warning(f"Failed to fetch geometry for {district_name}")
    
    if not district_data:
        logger.error("No district data was successfully fetched")
        return gpd.GeoDataFrame()
    
    # Create DataFrame and add geometries
    df = pd.DataFrame(district_data)
    df['geometry'] = df.apply(create_geometry_from_coords, axis=1)
    
    # Filter out failed geometries
    df = df[df['geometry'].notna()].reset_index(drop=True)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    
    logger.info(f"✅ Created district GeoDataFrame with {len(gdf)} districts")
    
    return gdf


# ===== SPATIAL VALIDATION =====
def validate_road_locations(
    roads_gdf: gpd.GeoDataFrame,
    districts_gdf: gpd.GeoDataFrame,
    district_col: str = 'district'
) -> gpd.GeoDataFrame:
    """
    Validate road locations against district boundaries using overlap length.
    
    Assigns each road to the district with maximum length overlap.
    
    Args:
        roads_gdf: GeoDataFrame of roads with LineString geometries
        districts_gdf: GeoDataFrame of district polygons
        district_col: Column name for district labels
    
    Returns:
        GeoDataFrame with validation columns added
    """
    logger.info(f"🔍 Validating {len(roads_gdf):,} road locations...")
    
    # Validation
    if roads_gdf.empty or districts_gdf.empty:
        logger.warning("Empty GeoDataFrame provided for validation")
        roads_gdf['actual_district'] = 'N/A'
        roads_gdf['is_in_correct_district'] = False
        return roads_gdf
    
    # Ensure matching CRS
    if roads_gdf.crs != districts_gdf.crs:
        logger.info(f"Reprojecting roads to match districts CRS: {districts_gdf.crs}")
        roads_gdf = roads_gdf.to_crs(districts_gdf.crs)
    
    # Fix any invalid geometries
    districts_clean = districts_gdf.copy()
    districts_clean['geometry'] = districts_clean['geometry'].buffer(0)
    
    # Validate each road
    results = []
    
    for idx, road in tqdm(roads_gdf.iterrows(), total=len(roads_gdf), desc="Validating"):
        road_geom = road.geometry
        assigned_district = road[district_col]
        
        best_district = None
        max_overlap = 0.0
        
        # Find district with maximum overlap
        for _, district in districts_clean.iterrows():
            if district.geometry.intersects(road_geom):
                intersection = district.geometry.intersection(road_geom)
                
                # Calculate intersection length
                if intersection.is_empty:
                    overlap_length = 0.0
                elif hasattr(intersection, 'length'):
                    overlap_length = intersection.length
                else:
                    # Handle MultiLineString or GeometryCollection
                    overlap_length = sum(
                        geom.length for geom in intersection.geoms
                        if hasattr(geom, 'length')
                    )
                
                if overlap_length > max_overlap:
                    max_overlap = overlap_length
                    best_district = district[district_col]
        
        # Build result
        record = road.to_dict()
        record['actual_district'] = best_district or 'Outside any district'
        record['is_in_correct_district'] = (assigned_district == best_district) if best_district else False
        record['overlap_length'] = max_overlap
        record['total_road_length'] = road_geom.length
        record['overlap_percentage'] = (max_overlap / road_geom.length * 100) if road_geom.length > 0 else 0
        
        results.append(record)
    
    # Create validated GeoDataFrame
    validated_gdf = gpd.GeoDataFrame(results, crs=roads_gdf.crs)
    
    # Summary statistics
    correct_count = validated_gdf['is_in_correct_district'].sum()
    correct_pct = correct_count / len(validated_gdf) * 100
    
    logger.info(f"✅ Validation complete:")
    logger.info(f"   • Correctly located: {correct_count:,}/{len(validated_gdf):,} ({correct_pct:.1f}%)")
    logger.info(f"   • Mismatched: {len(validated_gdf) - correct_count:,}")
    
    return validated_gdf


# ===== MAIN PIPELINE =====
def run_spatial_validation_pipeline(
    input_file: Path = CLEANED_DATA_DIR / "df_v1.csv",
    output_file: Path = CLEANED_DATA_DIR / "df_validated.csv",
    config: Optional[OSMConfig] = None
) -> pd.DataFrame:
    """
    Complete pipeline: extract geometries and validate spatial locations.
    
    Args:
        input_file: Path to input CSV with property data
        output_file: Path to save validated results
        config: OSM configuration (uses default if None)
    
    Returns:
        Validated DataFrame with spatial validation columns
    """
    if config is None:
        config = OSMConfig()
    
    logger.info("="*60)
    logger.info("SPATIAL VALIDATION PIPELINE STARTED")
    logger.info("="*60)
    
    # 1. Load geocoded data
    logger.info(f"📂 Loading data from '{input_file.name}'...")
    df = pd.read_csv(input_file)
    logger.info(f"   • Loaded {len(df):,} property records")
    
    # 2. Get unique roads with way_ids
    unique_roads = df[['road_name', 'way_id', 'district']].drop_duplicates()
    logger.info(f"   • Processing {len(unique_roads):,} unique roads")
    
    # 3. Extract road geometries
    roads_with_geom = extract_geometries_for_ways(
        unique_roads,
        id_column='way_id',
        name_column='road_name',
        config=config
    )
    
    # 4. Create district boundaries GeoDataFrame
    districts_gdf = create_district_geodataframe(DISTRICT_OSM_IDS, config)
    
    # 5. Validate road locations
    roads_gdf = gpd.GeoDataFrame(
        roads_with_geom[roads_with_geom['geometry'].notna()],
        crs="EPSG:4326"
    )
    
    validated_roads = validate_road_locations(
        roads_gdf,
        districts_gdf,
        district_col='district'
    )
    
    # 6. Merge back with original data
    logger.info("🔗 Merging validation results with original dataset...")
    validated_df = df.merge(
        validated_roads[['road_name', 'way_id', 'actual_district', 
                         'is_in_correct_district', 'overlap_length', 
                         'overlap_percentage']],
        on=['road_name', 'way_id'],
        how='left'
    )
    
    # 7. Save results
    validated_df.to_csv(output_file, index=False)
    logger.info(f"💾 Validated data saved to '{output_file.name}'")
    
    # 8. Final summary
    logger.info("\n" + "="*60)
    logger.info("📊 PIPELINE SUMMARY")
    logger.info("="*60)
    logger.info(f"   • Total records: {len(validated_df):,}")
    logger.info(f"   • Records with geometry: {validated_df['overlap_length'].notna().sum():,}")
    logger.info(f"   • Correctly located: {validated_df['is_in_correct_district'].sum():,}")
    logger.info("="*60)
    
    return validated_df


# ===== SCRIPT EXECUTION =====
if __name__ == "__main__":
    # Run the complete spatial validation pipeline
    validated_df = run_spatial_validation_pipeline()