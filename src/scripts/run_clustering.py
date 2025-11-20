#!/usr/bin/env python3
"""
Create market cluster segments using spatial clustering.

Corresponds to notebook 4 in the pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import load_interim_data, save_interim_data
from src.features import cluster_market_segments
from src.features.clustering import get_cluster_statistics
from src.config import ensure_directories, WITH_FEATURES_DATA, CLUSTERED_DATA

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Create market cluster segments using spatial clustering'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to feature data (default: data/interim/with_features.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save clustered data (default: data/interim/clustered.csv)'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['auto', 'dbscan', 'kmeans'],
        default='auto',
        help='Clustering method (default: auto)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Print cluster statistics'
    )

    args = parser.parse_args()

    ensure_directories()

    # Load feature data
    if args.input:
        logger.info(f"Loading data from {args.input}")
        from src.data import load_raw_data
        df = load_raw_data(args.input)
    else:
        logger.info("Loading feature data from interim directory")
        df = load_interim_data('with_features')

    # Perform clustering
    logger.info(f"Performing spatial clustering (method: {args.method})...")
    df_clustered = cluster_market_segments(df, method=args.method)

    # Print statistics if requested
    if args.stats:
        logger.info("Calculating cluster statistics...")
        stats = get_cluster_statistics(df_clustered)
        print("\n=== Cluster Statistics ===")
        print(stats.head(20))

    # Save results
    output_path = args.output if args.output else CLUSTERED_DATA
    logger.info(f"Saving clustered data to {output_path}")
    save_interim_data(df_clustered, output_path, stage_name="clustered")

    logger.info("✅ Clustering complete!")
    logger.info(f"Created {df_clustered['market_cluster_id'].nunique()} market segments")


if __name__ == '__main__':
    main()
