#!/usr/bin/env python3
"""
Train property price prediction models.

Corresponds to notebook 5 in the pipeline.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import load_interim_data
from src.features import preprocess_for_training
from src.features.preprocessing import clean_property_data, fill_missing_values
from src.models import train_all_models, compare_models
from src.models.trainer import save_model
from src.config import (
    ensure_directories,
    CLUSTERED_DATA,
    PRODUCTION_MODEL,
    SCALER_MODEL,
    MODEL_METADATA
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Train property price prediction models'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to clustered data (default: data/interim/clustered.csv)'
    )
    parser.add_argument(
        '--output-model',
        type=str,
        default=None,
        help='Path to save trained model (default: models/production_model_rf_pre2025.pkl)'
    )
    parser.add_argument(
        '--output-scaler',
        type=str,
        default=None,
        help='Path to save fitted scaler (default: models/scaler_pre2025.pkl)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=10,
        help='Number of cross-validation folds (default: 10)'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Specific models to train (default: all models)'
    )
    parser.add_argument(
        '--temporal-split',
        action='store_true',
        default=True,
        help='Use temporal validation (train on pre-2025, test on 2025)'
    )
    parser.add_argument(
        '--save-best',
        type=str,
        default='Random Forest',
        help='Model name to save as production model (default: Random Forest)'
    )

    args = parser.parse_args()

    ensure_directories()

    # Load clustered data
    if args.input:
        logger.info(f"Loading data from {args.input}")
        from src.data import load_raw_data
        df = load_raw_data(args.input)
    else:
        logger.info("Loading clustered data from interim directory")
        df = load_interim_data('clustered')

    # Clean data
    logger.info("Cleaning property data...")
    df_clean = clean_property_data(df)
    df_clean = fill_missing_values(df_clean)

    # Preprocess for training
    logger.info("Preprocessing data for model training...")
    preprocessed = preprocess_for_training(
        df_clean,
        temporal_split=args.temporal_split
    )

    # Train models
    logger.info(f"Training models with {args.cv_folds}-fold CV...")
    trained_models, cv_results = train_all_models(
        preprocessed['X_train'],
        preprocessed['y_train_log'],
        cv=args.cv_folds,
        models_to_train=args.models
    )

    # Compare models
    logger.info("Comparing model performance...")
    comparison = compare_models(
        trained_models,
        preprocessed['X_train'],
        preprocessed['X_test'],
        preprocessed['y_train_log'],
        preprocessed['y_test_log'],
        preprocessed['y_train'],
        preprocessed['y_test']
    )

    print("\n=== Model Comparison Results ===")
    print(comparison[['Ranking', 'Model', 'Test_Orig R2', 'Test_Orig MAPE']].to_string())

    # Save best model
    if args.save_best and args.save_best in trained_models:
        model_path = args.output_model if args.output_model else PRODUCTION_MODEL
        scaler_path = args.output_scaler if args.output_scaler else SCALER_MODEL

        logger.info(f"Saving {args.save_best} model...")
        save_model(trained_models[args.save_best], model_path, args.save_best)

        logger.info(f"Saving preprocessor/scaler...")
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(preprocessed['preprocessor'], f)

        # Save metadata
        logger.info("Saving model metadata...")
        best_metrics = comparison[comparison['Model'] == args.save_best].iloc[0]

        metadata = {
            "model_name": args.save_best,
            "training_timestamp": datetime.now().isoformat(),
            "n_train_samples": preprocessed['n_train'],
            "n_test_samples": preprocessed['n_test'],
            "n_features": preprocessed['n_features'],
            "cv_folds": args.cv_folds,
            "temporal_split": args.temporal_split,
            "metrics": {
                "test_r2": float(best_metrics['Test_Orig R2']),
                "test_mape": float(best_metrics['Test_Orig MAPE']),
                "test_rmse": float(best_metrics['Test_Orig RMSE']),
                "test_mae": float(best_metrics['Test_Orig MAE'])
            },
            "feature_names": preprocessed['feature_names']
        }

        with open(MODEL_METADATA, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✅ Model saved to {model_path}")
        logger.info(f"✅ Scaler saved to {scaler_path}")
        logger.info(f"✅ Metadata saved to {MODEL_METADATA}")

    logger.info("✅ Training complete!")


if __name__ == '__main__':
    main()
