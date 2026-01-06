"""
Configuration loader.
Place this file in the src/ directory and import it.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Find the .env file (in parent directory)
parent_dir = Path(__file__).parent
project_root = parent_dir.parent
env_file = project_root / '.env'

# Load environment variables
load_dotenv(dotenv_path=env_file)

# Export configuration variables
DATA_RAW_DIR = Path(os.getenv('DATA_RAW_DIR', ''))
CLEANED_DATA_DIR = Path(os.getenv('CLEANED_DATA_DIR', ''))
SUPPORT_DATA_DIR = Path(os.getenv('SUPPORT_DATA_DIR', ''))
DATA_PROCESSED_DIR = Path(os.getenv('DATA_PROCESSED_DIR', ''))
VISUALIZATIONS_DIR = Path(os.getenv('VISUALIZATIONS_DIR', ''))
MODELS_DIR = Path(os.getenv('MODELS_DIR', ''))

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