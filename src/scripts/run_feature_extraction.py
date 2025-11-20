#!/usr/bin/env python3
"""
Extract geospatial features using OpenStreetMap.

Corresponds to notebooks 2.1, 2.2, and 3 in the pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import load_interim_data, save_interim_data
from src.features import extract_amenity_features, extract_transit_ridership
from src.features.geospatial import fill_missing_distances
from src.config import ensure_directories, GEOCODED_DATA, WITH_FEATURES_DATA

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Extract geospatial features from OpenStreetMap'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to geocoded data (default: data/interim/geocoded.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save feature data (default: data/interim/with_features.csv)'
    )
    parser.add_argument(
        '--skip-amenities',
        action='store_true',
        help='Skip amenity feature extraction'
    )
    parser.add_argument(
        '--skip-ridership',
        action='store_true',
        help='Skip transit ridership extraction'
    )
    parser.add_argument(
        '--ridership-data',
        type=str,
        default=None,
        help='Path to ridership CSV file'
    )

    args = parser.parse_args()

    ensure_directories()

    # Load geocoded data
    if args.input:
        logger.info(f"Loading data from {args.input}")
        df = load_raw_data(args.input)
    else:
        logger.info("Loading geocoded data from interim directory")
        df = load_interim_data('geocoded')

    # Extract amenity features
    if not args.skip_amenities:
        logger.info("Extracting amenity features...")
        df = extract_amenity_features(df)

    # Extract transit ridership
    if not args.skip_ridership:
        logger.info("Extracting transit ridership features...")
        df = extract_transit_ridership(df, args.ridership_data)

    # Fill missing distances
    logger.info("Filling missing distance values...")
    df = fill_missing_distances(df)

    # Save results
    output_path = args.output if args.output else WITH_FEATURES_DATA
    logger.info(f"Saving feature data to {output_path}")
    save_interim_data(df, output_path, stage_name="with_features")

    logger.info("✅ Feature extraction complete!")


if __name__ == '__main__':
    main()
