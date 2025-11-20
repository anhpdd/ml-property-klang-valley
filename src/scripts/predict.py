#!/usr/bin/env python3
"""
Make predictions on new property data using trained model.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import PropertyPredictor, load_and_predict
from src.config import PRODUCTION_MODEL, SCALER_MODEL

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Predict property prices using trained model'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input data file (CSV or Excel) with property features'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save predictions (default: adds predictions to input file)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help=f'Path to trained model (default: {PRODUCTION_MODEL})'
    )
    parser.add_argument(
        '--scaler',
        type=str,
        default=None,
        help=f'Path to fitted scaler (default: {SCALER_MODEL})'
    )
    parser.add_argument(
        '--price-col',
        type=str,
        default='predicted_price_m2',
        help='Column name for predictions (default: predicted_price_m2)'
    )

    args = parser.parse_args()

    # Load and predict
    logger.info(f"Loading data and making predictions...")
    results = load_and_predict(
        args.input,
        model_path=args.model,
        scaler_path=args.scaler,
        output_path=args.output
    )

    # Print summary
    logger.info("\n=== Prediction Summary ===")
    logger.info(f"Total properties: {len(results)}")
    logger.info(f"Mean predicted price: RM {results['predicted_price_m2'].mean():,.2f}/m²")
    logger.info(f"Median predicted price: RM {results['predicted_price_m2'].median():,.2f}/m²")
    logger.info(f"Min predicted price: RM {results['predicted_price_m2'].min():,.2f}/m²")
    logger.info(f"Max predicted price: RM {results['predicted_price_m2'].max():,.2f}/m²")

    if args.output:
        logger.info(f"✅ Predictions saved to {args.output}")
    else:
        logger.info("✅ Predictions complete!")
        print("\nFirst 10 predictions:")
        print(results[['predicted_price_m2']].head(10))


if __name__ == '__main__':
    main()
