#!/usr/bin/env python3
"""
Geocode property addresses using OpenStreetMap.

Corresponds to notebooks 0, 1.1, and 1.2 in the pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import load_raw_data, save_interim_data, geocode_properties, validate_coordinates
from src.config import ensure_directories, GEOCODED_DATA

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Geocode property addresses using OpenStreetMap'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input data file (CSV or Excel)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save geocoded data (default: data/interim/geocoded.csv)'
    )
    parser.add_argument(
        '--road-col',
        type=str,
        default='road_name',
        help='Column name containing road names (default: road_name)'
    )
    parser.add_argument(
        '--district-col',
        type=str,
        default='district',
        help='Column name containing district names (default: district)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate coordinates after geocoding'
    )

    args = parser.parse_args()

    # Ensure directories exist
    ensure_directories()

    # Load data
    logger.info(f"Loading data from {args.input}")
    df = load_raw_data(args.input)

    # Geocode properties
    logger.info("Starting geocoding...")
    df_geocoded = geocode_properties(
        df,
        road_col=args.road_col,
        district_col=args.district_col
    )

    # Validate coordinates if requested
    if args.validate:
        logger.info("Validating coordinates...")
        df_geocoded = validate_coordinates(df_geocoded)

    # Save results
    output_path = args.output if args.output else GEOCODED_DATA
    logger.info(f"Saving geocoded data to {output_path}")
    save_interim_data(df_geocoded, output_path, stage_name="geocoded")

    logger.info("✅ Geocoding complete!")


if __name__ == '__main__':
    main()
