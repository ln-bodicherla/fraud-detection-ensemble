"""
Command-line interface for fraud detection predictions.

Loads a trained model and generates fraud probability scores
on new claims data from CSV or Parquet input files.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import FraudDetectionModel

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate fraud predictions on new claims data"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input claims file (CSV or Parquet)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="predictions.csv",
        help="Path to output predictions file (default: predictions.csv)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/",
        help="Directory containing saved model artifacts",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for binary fraud classification (default: 0.5)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        choices=["xgboost", "lightgbm", "catboost", None],
        help="Specific model to use (default: stacking ensemble)",
    )
    parser.add_argument(
        "--include-features",
        action="store_true",
        help="Include engineered features in output",
    )
    parser.add_argument(
        "--top-risk",
        type=int,
        default=None,
        help="Output only the top N highest-risk claims",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def load_input_data(input_path: str) -> pd.DataFrame:
    """Load claims data from the specified file path."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    logger.info("Loaded %d claims from %s", len(df), input_path)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing and feature engineering to input data."""
    preprocessor = DataPreprocessor()
    df = preprocessor.handle_missing_values(df)

    has_target = "is_fraud" in df.columns
    if has_target:
        target = df["is_fraud"].copy()

    df = preprocessor.encode_categoricals(df)

    engineer = FeatureEngineer()
    df = engineer.build_feature_matrix(df)

    if has_target and "is_fraud" not in df.columns:
        df["is_fraud"] = target

    return df


def generate_predictions(
    model: FraudDetectionModel,
    df: pd.DataFrame,
    model_name: str = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Generate fraud predictions and add them to the DataFrame."""
    exclude_cols = ["is_fraud", "claim_id"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].select_dtypes(include=[np.number])

    probabilities = model.predict(X, model_name=model_name)
    binary_predictions = (probabilities >= threshold).astype(int)

    risk_categories = pd.cut(
        probabilities,
        bins=[0.0, 0.2, 0.5, 0.8, 1.0],
        labels=["low", "medium", "high", "critical"],
        include_lowest=True,
    )

    df["fraud_probability"] = np.round(probabilities, 6)
    df["fraud_prediction"] = binary_predictions
    df["risk_category"] = risk_categories

    return df


def format_output(
    df: pd.DataFrame,
    include_features: bool = False,
    top_risk: int = None,
) -> pd.DataFrame:
    """Format the output DataFrame for saving."""
    if not include_features:
        output_cols = ["claim_id", "fraud_probability", "fraud_prediction", "risk_category"]
        available = [c for c in output_cols if c in df.columns]
        if not available:
            available = ["fraud_probability", "fraud_prediction", "risk_category"]
        df = df[available]

    df = df.sort_values("fraud_probability", ascending=False)

    if top_risk is not None:
        df = df.head(top_risk)

    return df


def save_predictions(df: pd.DataFrame, output_path: str) -> None:
    """Save predictions to the output file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

    logger.info("Predictions saved to %s (%d claims)", output_path, len(df))


def print_summary(df: pd.DataFrame, threshold: float) -> None:
    """Print a summary of prediction results."""
    total = len(df)
    flagged = (df["fraud_prediction"] == 1).sum()
    flag_rate = flagged / total if total > 0 else 0

    risk_dist = df["risk_category"].value_counts()

    print("\n" + "=" * 50)
    print("FRAUD PREDICTION SUMMARY")
    print("=" * 50)
    print(f"Total claims scored:  {total:,}")
    print(f"Flagged as fraud:     {flagged:,} ({flag_rate:.1%})")
    print(f"Decision threshold:   {threshold}")
    print(f"\nMean fraud probability:   {df['fraud_probability'].mean():.4f}")
    print(f"Median fraud probability: {df['fraud_probability'].median():.4f}")
    print(f"Max fraud probability:    {df['fraud_probability'].max():.4f}")
    print(f"\nRisk Category Distribution:")
    for category in ["critical", "high", "medium", "low"]:
        count = risk_dist.get(category, 0)
        print(f"  {category:>10s}: {count:>6,} ({count / total:.1%})")
    print("=" * 50)


def main() -> None:
    """Main entry point for the prediction CLI."""
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Loading model from %s", args.model_dir)
    model = FraudDetectionModel(config_path=args.config)

    model_dir = Path(args.model_dir)
    if model_dir.exists():
        model.load_models(args.model_dir)
    else:
        logger.warning(
            "Model directory %s not found. Train models first using "
            "FraudDetectionModel.train_stacking_ensemble()",
            args.model_dir,
        )
        sys.exit(1)

    df = load_input_data(args.input)
    df = preprocess_data(df)

    df = generate_predictions(
        model=model,
        df=df,
        model_name=args.model_name,
        threshold=args.threshold,
    )

    output_df = format_output(
        df,
        include_features=args.include_features,
        top_risk=args.top_risk,
    )

    save_predictions(output_df, args.output)
    print_summary(df, args.threshold)


if __name__ == "__main__":
    main()
