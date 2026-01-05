"""
Data handling modules for property valuation analysis.

This package provides a complete data pipeline for:
1. Loading and cleaning residential property data
2. Geocoding property addresses using OpenStreetMap
3. Extracting road geometries and validating spatial locations

Quick Start
-----------
    from src.data import load_and_clean_data, run_geocoding_pipeline, run_spatial_validation_pipeline
    
    # Step 1: Load and clean raw data
    clean_df = load_and_clean_data()
    
    # Step 2: Geocode addresses
    geocoded_df = run_geocoding_pipeline()
    
    # Step 3: Validate spatial locations
    validated_df = run_spatial_validation_pipeline()

Custom Workflows
----------------
    from src.data import geocode_unique_roads, GeocodingConfig
    from src.data import validate_road_locations, OSMConfig
    
    # Custom geocoding with different rate limits
    config = GeocodingConfig(min_delay_seconds=2.0)
    road_to_way_id, road_to_node_info = geocode_unique_roads(
        unique_roads, config
    )
    
    # Custom spatial validation
    validated_gdf = validate_road_locations(roads_gdf, districts_gdf)
"""

# ===== LOADERS MODULE =====
from .loaders import (
    # Main pipeline
    load_and_clean_data,
    
    # Data loading
    load_yearly_data,
    
    # Individual cleaning functions (for custom workflows)
    clean_column_names,
    clean_numeric_columns,
    swap_mismatched_areas,
    clean_and_process_addresses,
    clean_unit_level,
)

# ===== GEOCODING MODULE =====
from .geocoding import (
    # Main pipeline
    run_geocoding_pipeline,
    
    # Configuration
    GeocodingConfig,
    
    # Core functions (for custom workflows)
    setup_geocoder,
    geocode_unique_roads,
    geocode_single_road,
    enrich_dataframe_with_geocodes,
    calculate_geocoding_stats,
    save_geocoding_results,
    
    # Checkpoint management
    load_checkpoint,
    save_checkpoint,
)

# ===== SPATIAL VALIDATION MODULE =====
from .validation import (
    # Main pipeline
    run_spatial_validation_pipeline,
    
    # Configuration
    OSMConfig,
    
    # Geometry extraction
    extract_geometries_for_ways,
    fetch_way_geometry,
    extract_line_from_way,
    
    # District boundaries
    create_district_geodataframe,
    DISTRICT_OSM_IDS,
    
    # Spatial validation
    validate_road_locations,
    
    # Complex polygon processing
    stitch_relation_polygons,
    fetch_relation_geometry,
    create_geometry_from_coords,
    
    # Caching
    OSMCache,
)


# ===== PUBLIC API =====
__all__ = [
    # === MAIN PIPELINES (Most Common Usage) ===
    'load_and_clean_data',
    'run_geocoding_pipeline',
    'run_spatial_validation_pipeline',
    
    # === CONFIGURATION CLASSES ===
    'GeocodingConfig',
    'OSMConfig',
    
    # === DATA LOADING ===
    'load_yearly_data',
    'clean_column_names',
    'clean_numeric_columns',
    'swap_mismatched_areas',
    'clean_and_process_addresses',
    'clean_unit_level',
    
    # === GEOCODING ===
    'setup_geocoder',
    'geocode_unique_roads',
    'geocode_single_road',
    'enrich_dataframe_with_geocodes',
    'calculate_geocoding_stats',
    'save_geocoding_results',
    'load_checkpoint',
    'save_checkpoint',
    
    # === SPATIAL VALIDATION ===
    'extract_geometries_for_ways',
    'fetch_way_geometry',
    'extract_line_from_way',
    'create_district_geodataframe',
    'validate_road_locations',
    'stitch_relation_polygons',
    'fetch_relation_geometry',
    'create_geometry_from_coords',
    'OSMCache',
    
    # === CONSTANTS ===
    'DISTRICT_OSM_IDS',
]


# ===== VERSION INFO =====
__version__ = '1.0.0'
__author__ = 'Duy Anh'
__description__ = 'Property valuation data pipeline for Shah Alam LRT3 corridor analysis'