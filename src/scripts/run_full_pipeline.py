#!/usr/bin/env python3
"""
Run the complete property valuation pipeline end-to-end.

Executes all stages: geocoding → feature extraction → clustering → model training
"""

import argparse
import logging
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    ensure_directories,
    GEOCODED_DATA,
    WITH_FEATURES_DATA,
    CLUSTERED_DATA,
    PRODUCTION_MODEL
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(script_name: str, args: list = None) -> int:
    """
    Run a pipeline script as subprocess.

    Args:
        script_name: Name of script to run
        args: Additional arguments for the script

    Returns:
        int: Return code
    """
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description='Run complete property valuation pipeline end-to-end'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to raw input data file (CSV or Excel)'
    )
    parser.add_argument(
        '--skip-geocoding',
        action='store_true',
        help='Skip geocoding stage (use existing geocoded data)'
    )
    parser.add_argument(
        '--skip-features',
        action='store_true',
        help='Skip feature extraction stage'
    )
    parser.add_argument(
        '--skip-clustering',
        action='store_true',
        help='Skip clustering stage'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training stage'
    )
    parser.add_argument(
        '--clustering-method',
        type=str,
        choices=['auto', 'dbscan', 'kmeans'],
        default='auto',
        help='Clustering method (default: auto)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=10,
        help='Cross-validation folds for training (default: 10)'
    )

    args = parser.parse_args()

    # Ensure directories exist
    ensure_directories()

    scripts_dir = Path(__file__).parent

    # Stage 1: Geocoding
    if not args.skip_geocoding:
        logger.info("\n" + "="*70)
        logger.info("STAGE 1: GEOCODING")
        logger.info("="*70)

        returncode = run_command(
            str(scripts_dir / 'run_geocoding.py'),
            ['--input', args.input, '--validate']
        )

        if returncode != 0:
            logger.error("❌ Geocoding failed! Aborting pipeline.")
            return 1
    else:
        logger.info("⏭️  Skipping geocoding stage")

    # Stage 2: Feature Extraction
    if not args.skip_features:
        logger.info("\n" + "="*70)
        logger.info("STAGE 2: FEATURE EXTRACTION")
        logger.info("="*70)

        returncode = run_command(
            str(scripts_dir / 'run_feature_extraction.py'),
            []
        )

        if returncode != 0:
            logger.error("❌ Feature extraction failed! Aborting pipeline.")
            return 1
    else:
        logger.info("⏭️  Skipping feature extraction stage")

    # Stage 3: Clustering
    if not args.skip_clustering:
        logger.info("\n" + "="*70)
        logger.info("STAGE 3: SPATIAL CLUSTERING")
        logger.info("="*70)

        returncode = run_command(
            str(scripts_dir / 'run_clustering.py'),
            ['--method', args.clustering_method, '--stats']
        )

        if returncode != 0:
            logger.error("❌ Clustering failed! Aborting pipeline.")
            return 1
    else:
        logger.info("⏭️  Skipping clustering stage")

    # Stage 4: Model Training
    if not args.skip_training:
        logger.info("\n" + "="*70)
        logger.info("STAGE 4: MODEL TRAINING")
        logger.info("="*70)

        returncode = run_command(
            str(scripts_dir / 'train_model.py'),
            ['--cv-folds', str(args.cv_folds), '--temporal-split']
        )

        if returncode != 0:
            logger.error("❌ Model training failed! Aborting pipeline.")
            return 1
    else:
        logger.info("⏭️  Skipping model training stage")

    # Pipeline complete
    logger.info("\n" + "="*70)
    logger.info("✅ PIPELINE COMPLETE!")
    logger.info("="*70)
    logger.info(f"Geocoded data: {GEOCODED_DATA}")
    logger.info(f"Feature data: {WITH_FEATURES_DATA}")
    logger.info(f"Clustered data: {CLUSTERED_DATA}")
    logger.info(f"Trained model: {PRODUCTION_MODEL}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
