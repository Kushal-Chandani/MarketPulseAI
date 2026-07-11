"""Backward-compatible training entrypoint for logistic regression."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import load_feature_frame, persist_artifacts, split_time_series
from .common import train_and_evaluate as _train_and_evaluate
from .train_logistic import build_model

DEFAULT_MODEL_PATH = Path("artifacts/model.joblib")
DEFAULT_METRICS_PATH = Path("artifacts/metrics.json")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the default training command."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate the default logistic regression model on XAUUSD features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/xauusd_1h_features.csv"),
        help="Processed feature CSV path.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Model artifact path. Defaults to {DEFAULT_MODEL_PATH}.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help=f"Metrics JSON path. Defaults to {DEFAULT_METRICS_PATH}.",
    )
    return parser.parse_args()


def train_and_evaluate(frame):
    """Backward-compatible wrapper around the logistic regression trainer."""
    return _train_and_evaluate(frame, build_model(), "logistic_regression")


def main() -> None:
    """Run the default logistic regression training command."""
    args = parse_args()
    frame = load_feature_frame(args.input)
    model, metrics = train_and_evaluate(frame)
    persist_artifacts(model, metrics, args.model_output, args.metrics_output)
    print(
        "Saved trained model to "
        f"{args.model_output} and evaluation metrics to {args.metrics_output}."
    )


if __name__ == "__main__":
    main()
