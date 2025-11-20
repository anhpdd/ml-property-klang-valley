"""
Configuration file for ML Property Valuation project.

Contains all constants, paths, feature definitions, and hyperparameters
used throughout the codebase.
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ============================================================================
# DATA FILE PATHS
# ============================================================================

# Interim data files (outputs from each pipeline stage)
GEOCODED_DATA = DATA_INTERIM_DIR / "geocoded.csv"
WITH_FEATURES_DATA = DATA_INTERIM_DIR / "with_features.csv"
MERGED_DATA = DATA_INTERIM_DIR / "merged.csv"
CLUSTERED_DATA = DATA_INTERIM_DIR / "clustered.csv"

# Processed data (final training-ready data)
PROCESSED_DATA = DATA_PROCESSED_DIR / "ready_for_model.csv"

# Model files
PRODUCTION_MODEL = MODELS_DIR / "production_model_rf_pre2025.pkl"
SCALER_MODEL = MODELS_DIR / "scaler_pre2025.pkl"
MODEL_METADATA = MODELS_DIR / "model_metadata.json"

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Color palette (consistent across all visualizations)
COLORS = {
    'blue': '#174A7E',
    'orange': '#FAA43A',
    'red': '#F15854',
    'grey': '#808080',
    'green': '#2ECC71'
}

# Plot settings
FIGURE_SIZE = (12, 6)
DPI = 300
PLOT_STYLE = "white"

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

# Distance-based features
DISTANCE_COLS = [
    'walk_dist_to_mall',
    'walk_dist_to_school',
    'walk_dist_to_park',
    'walk_dist_to_river',
    'walk_dist_to_lake',
    'walk_dist_to_rail_station',
    'dist_to_mall',
    'dist_to_school',
    'dist_to_park',
    'dist_to_river',
    'dist_to_lake',
    'dist_to_rail_station'
]

# Count-based features (amenities within 1km radius)
COUNT_COLS = [
    'mall_count',
    'school_count',
    'park_count',
    'river_count',
    'lake_count',
    'rail_station_count'
]

# Property attribute features
PROPERTY_COLS = [
    'property_m2',
    'unit_level'
]

# Transit ridership features
RIDERSHIP_COLS = [
    'incoming_ridership_within_1km',
    'outgoing_ridership_within_1km'
]

# All continuous features (for scaling)
CONTINUOUS_FEATURES = PROPERTY_COLS + DISTANCE_COLS + COUNT_COLS + RIDERSHIP_COLS

# Categorical features (for one-hot encoding)
CATEGORICAL_FEATURES = [
    'property_type',
    'market_cluster_id',
    'freehold',
    'transit',
    'district'
]

# Columns to drop during preprocessing
COLUMNS_TO_DROP = [
    'records_id',
    'geometry',
    'x_coord',
    'y_coord',
    'way_id',
    'train_ids',
    'Unnamed: 0',  # Redundant IDs or geographic coordinates
    'date',  # Year already extracted
    'mukim',
    'road_name',
    'scheme_name',  # market_cluster_id represents these
    'market_cluster',
    'clustering_method',
    'is_noise',  # Redundant cluster information
    'transaction_price',
    'land_m2',  # Removed due to new price_m2 creation
    'total_ridership_within_1km'  # Redundant ridership information
]

# Target variable
TARGET_VARIABLE = 'price_m2'

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# Random seed for reproducibility
RANDOM_STATE = 42

# Train/test split
TEST_SIZE = 0.3

# Cross-validation
CV_FOLDS = 10

# Temporal split year threshold
TEMPORAL_SPLIT_YEAR = 2025  # Train on pre-2025, test on 2025

# Missing value fill
MAX_DISTANCE_KM = 21  # Cap distances at 21 km if district max is missing

# ============================================================================
# CLUSTERING PARAMETERS
# ============================================================================

# DBSCAN parameters
DBSCAN_EPS_PERCENTILE_MIN = 0.90
DBSCAN_EPS_PERCENTILE_MAX = 0.95
DBSCAN_MIN_SAMPLES = 5
DBSCAN_METRIC = 'haversine'
DBSCAN_MAX_NOISE_THRESHOLD = 0.15  # Fallback to K-Means if >15% noise

# K-Means parameters (fallback)
KMEANS_N_CLUSTERS = 238

# Cluster ID prefixes by district
CLUSTER_PREFIXES = {
    'gombak': 'GO',
    'hulu langat': 'HU',
    'hulu selangor': 'HU',
    'kuala lumpur': 'KL',
    'kuala langat': 'KU',
    'kuala selangor': 'KU',
    'petaling': 'PE',
    'putrajaya': 'PU',
    'sabak bernam': 'SA',
    'sepang': 'SE',
    'klang': 'KL'
}

# ============================================================================
# OPENSTREETMAP (OSM) SETTINGS
# ============================================================================

# OSM API settings
OSM_API_TIMEOUT = 180  # seconds
OSM_API_RATE_LIMIT = 1.0  # seconds between requests
OSM_MAX_RETRIES = 3
OSM_RETRY_BACKOFF = 2.0  # exponential backoff multiplier

# District OSM IDs (Malaysia administrative boundaries)
DISTRICT_OSM_IDS = {
    'gombak': 'R4575878',
    'hulu langat': 'R4575885',
    'hulu selangor': 'R4575886',
    'kuala lumpur': 'R4575874',
    'kuala langat': 'R4575883',
    'kuala selangor': 'R4575884',
    'petaling': 'R4575879',
    'putrajaya': 'R2118712',
    'sabak bernam': 'R4575881',
    'sepang': 'R4575880',
    'klang': 'R4575882'
}

# OSM amenity tags
OSM_AMENITY_TAGS = {
    'school': {'amenity': 'school'},
    'mall': {'shop': 'mall'},
    'park': {'leisure': 'park'},
    'river': {'waterway': 'river'},
    'lake': {'natural': 'water'},
    'rail_station': {'railway': 'station'}
}

# Search radius for amenity counts (km)
AMENITY_SEARCH_RADIUS_KM = 1.0

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_LEVEL = 'INFO'

# ============================================================================
# MODEL REGISTRY
# ============================================================================

# Available models for training
AVAILABLE_MODELS = [
    "Linear Regression",
    "Ridge",
    "Lasso",
    "KNeighbors Regressor",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "LightGBM"
]

# Production model
PRODUCTION_MODEL_NAME = "Random Forest"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directories():
    """Create all required directories if they don't exist."""
    dirs = [
        DATA_DIR,
        DATA_RAW_DIR,
        DATA_INTERIM_DIR,
        DATA_PROCESSED_DIR,
        MODELS_DIR,
        VISUALIZATIONS_DIR
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_feature_lists():
    """
    Get all feature lists as a dictionary.

    Returns:
        dict: Dictionary containing all feature lists
    """
    return {
        'distance_cols': DISTANCE_COLS,
        'count_cols': COUNT_COLS,
        'property_cols': PROPERTY_COLS,
        'ridership_cols': RIDERSHIP_COLS,
        'continuous_features': CONTINUOUS_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES
    }
