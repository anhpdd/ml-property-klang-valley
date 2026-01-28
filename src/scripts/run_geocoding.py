#!/usr/bin/env python3
"""
Geocode property addresses using OpenStreetMap.

Corresponds to notebooks 0, 1.1, and 1.2 in the pipeline.
"""

import argparse
import logging
from pathlib import Path

# Add parent directory to path

from src.data import run_geocoding_pipeline, run_spatial_validation_pipeline
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
        '--validate',
        action='store_true',
        help='Validate coordinates after geocoding'
    )

    args = parser.parse_args()

    # Ensure directories exist
    ensure_directories()

    # Geocode properties
    logger.info("Starting geocoding pipeline...")
    output_path = Path(args.output) if args.output else GEOCODED_DATA

    df_geocoded = run_geocoding_pipeline(
        input_file=Path(args.input),
        output_file=output_path
    )

    # Validate coordinates if requested
    if args.validate:
        logger.info("Starting spatial validation pipeline...")
        df_geocoded = run_spatial_validation_pipeline(
            input_file=output_path,
            output_file=output_path
        )

    logger.info("✅ Geocoding and validation complete!")


if __name__ == '__main__':
    main()
