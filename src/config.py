"""
Configuration loader.
Place this file in the src/ directory and import it.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Find the .env file (in parent directory)
parent_dir = Path(__file__).parent
PROJECT_ROOT = parent_dir.parent
env_file = PROJECT_ROOT / '.env'

# Load environment variables
load_dotenv(dotenv_path=env_file)

# Export configuration variables for directories
DATA_RAW_DIR = Path(os.getenv('DATA_RAW_DIR', 'data/raw'))
CLEANED_DATA_DIR = Path(os.getenv('CLEANED_DATA_DIR', 'data/interim'))
SUPPORT_DATA_DIR = Path(os.getenv('SUPPORT_DATA_DIR', 'data/support'))
DATA_PROCESSED_DIR = Path(os.getenv('DATA_PROCESSED_DIR', 'data/processed'))
VISUALIZATIONS_DIR = Path(os.getenv('VISUALIZATIONS_DIR', 'visualizations'))
MODELS_DIR = Path(os.getenv('MODELS_DIR', 'models'))

# File paths
GEOCODED_DATA = CLEANED_DATA_DIR / "geocoded.csv"
WITH_FEATURES_DATA = CLEANED_DATA_DIR / "with_features.csv"
CLUSTERED_DATA = CLEANED_DATA_DIR / "clustered.csv"
PRODUCTION_MODEL = MODELS_DIR / "production_model_rf_pre2025.pkl"
SCALER_MODEL = MODELS_DIR / "scaler_pre2025.pkl"
MODEL_METADATA = MODELS_DIR / "model_metadata.json"

# Colors for visualization
COLORS = {
    'blue': '#174A7E',
    'orange': '#FAA43A',
    'red': '#F15854',
    'grey': '#808080',
    'green': '#2ECC71'
}

# District OSM IDs for Klang Valley
DISTRICT_OSM_IDS = {
    'GOMBAK': '12438352',
    'HULU LANGAT': '12438351',
    'KLANG': '12391135',
    'HULU SELANGOR': '10714199',
    'KUALA LANGAT': '10743362',
    'KUALA LUMPUR': '2939672',
    'KUALA SELANGOR': '10714137',
    'PETALING': '12391134',
    'PUTRAJAYA': '4443881',
    'SABAK BERNAM': '10714136',
    'SEPANG': '10743315'
}

# POI categories to extract from OSM
OSM_AMENITY_TAGS = {
    'school': {'amenity': 'school'},
    'mall': {'shop': 'supermarket'},
    'park': {'leisure': 'park'},
    'river': {'waterway': 'river'},
    'lake': {'natural': 'water', 'water': 'lake'}
}

# RapidKL train lines (OSM relation IDs)
TRAIN_LINES = {
    'LRT3 Shah Alam': 8394085,
    'LRT Kelana Jaya': 8000438,
    'LRT Sri Petaling': 8000387,
    'LRT Ampang': 8000297,
    'MRT Kajang': 5690837,
    'MRT Putrajaya': 11313577,
    'BRT Sunway Line': 11549145
}

# Model and Pipeline settings
RANDOM_STATE = 42
CV_FOLDS = 10
TEST_SIZE = 0.2
AMENITY_SEARCH_RADIUS_KM = 1.0
FIGURE_SIZE = (12, 6)
DPI = 300
PLOT_STYLE = 'white'
TEMPORAL_SPLIT_YEAR = 2025
TARGET_VARIABLE = 'price_m2'
MAX_DISTANCE_KM = 21.0

# Columns to drop during preprocessing
COLUMNS_TO_DROP = [
    'records_id', 'geometry', 'x_coord', 'y_coord', 'way_id',
    'train_ids', 'Unnamed: 0', 'date', 'mukim', 'road_name',
    'scheme_name', 'market_cluster', 'clustering_method',
    'is_noise', 'transaction_price', 'land_m2',
    'total_ridership_within_1km'
]

# Clustering settings
DBSCAN_EPS_PERCENTILE_MIN = 0.90
DBSCAN_EPS_PERCENTILE_MAX = 0.95
DBSCAN_MIN_SAMPLES = 5
DBSCAN_METRIC = 'haversine'
DBSCAN_MAX_NOISE_THRESHOLD = 0.15
KMEANS_N_CLUSTERS = 238

# District Prefixes for Cluster IDs
CLUSTER_PREFIXES = {
    'gombak': 'GO',
    'hulu langat': 'HU',
    'hulu selangor': 'HU',
    'klang': 'KL',
    'kuala langat': 'KU',
    'kuala selangor': 'KU',
    'kuala lumpur': 'KL',
    'petaling': 'PE',
    'putrajaya': 'PU',
    'sabak bernam': 'SA',
    'sepang': 'SE'
}

# Security settings
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB limit

# OSM API settings
OSM_API_TIMEOUT = 10
OSM_API_RATE_LIMIT = 1.1
OSM_MAX_RETRIES = 3
OSM_RETRY_BACKOFF = 2.0

# Feature lists
DISTANCE_COLS = [
    'walk_dist_to_mall', 'walk_dist_to_school', 'walk_dist_to_park',
    'walk_dist_to_river', 'walk_dist_to_lake', 'walk_dist_to_rail_station',
    'dist_to_mall', 'dist_to_school', 'dist_to_park',
    'dist_to_river', 'dist_to_lake', 'dist_to_rail_station'
]

COUNT_COLS = [
    'mall_count', 'school_count', 'park_count',
    'river_count', 'lake_count', 'rail_station_count'
]

RIDERSHIP_COLS = [
    'incoming_ridership_within_1km', 'outgoing_ridership_within_1km'
]

PROPERTY_COLS = [
    'property_m2', 'unit_level'
]

CONTINUOUS_FEATURES = DISTANCE_COLS + COUNT_COLS + RIDERSHIP_COLS + PROPERTY_COLS

CATEGORICAL_FEATURES = ['property_type', 'district', 'market_cluster_id']

AVAILABLE_MODELS = [
    "Linear Regression", "Ridge", "Lasso", "KNeighbors Regressor",
    "Decision Tree", "Random Forest", "Gradient Boosting",
    "XGBoost", "LightGBM"
]


def ensure_directories():
    """Ensure all required project directories exist."""
    directories = [
        DATA_RAW_DIR, CLEANED_DATA_DIR, SUPPORT_DATA_DIR,
        DATA_PROCESSED_DIR, VISUALIZATIONS_DIR, MODELS_DIR
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")


def get_feature_lists():
    """Returns a dictionary containing all feature lists."""
    return {
        'distance_cols': DISTANCE_COLS,
        'count_cols': COUNT_COLS,
        'ridership_cols': RIDERSHIP_COLS,
        'property_cols': PROPERTY_COLS,
        'continuous_features': CONTINUOUS_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES
    }
