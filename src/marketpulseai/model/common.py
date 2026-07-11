"""Shared training and evaluation utilities for model experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from marketpulseai.features.build import FEATURE_COLUMNS, TARGET_COLUMN

DEFAULT_INPUT_PATH = Path("data/processed/xauusd_1h_features.csv")


def load_feature_frame(input_path: Path) -> pd.DataFrame:
    """Load the engineered feature dataset from disk."""
    frame = pd.read_csv(input_path, parse_dates=["timestamp"])
    validate_feature_frame(frame)
    return frame.sort_values("timestamp").reset_index(drop=True)


def validate_feature_frame(frame: pd.DataFrame) -> None:
    """Ensure the engineered dataset contains the required columns."""
    required_columns = ["timestamp", *FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise RuntimeError(f"Feature frame is missing required columns: {missing_columns}")
    if frame.empty:
        raise RuntimeError("Feature frame is empty.")
    if frame["timestamp"].duplicated().any():
        raise RuntimeError("Feature frame contains duplicate timestamps.")
    if not frame["timestamp"].is_monotonic_increasing:
        raise RuntimeError("Feature frame timestamps are not sorted ascending.")


def split_time_series(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into chronological train, validation, and test windows."""
    total_rows = len(frame)
    train_end = int(total_rows * train_fraction)
    validation_end = train_end + int(total_rows * validation_fraction)

    train_frame = frame.iloc[:train_end].copy()
    validation_frame = frame.iloc[train_end:validation_end].copy()
    test_frame = frame.iloc[validation_end:].copy()

    if train_frame.empty or validation_frame.empty or test_frame.empty:
        raise RuntimeError("Time-based split produced an empty partition.")

    return train_frame, validation_frame, test_frame


def evaluate_split(
    model: Any,
    frame: pd.DataFrame,
    split_name: str,
) -> dict[str, Any]:
    """Evaluate a fitted model on a single chronological split."""
    features = frame[FEATURE_COLUMNS]
    target = frame[TARGET_COLUMN]
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()

    return {
        "split": split_name,
        "rows": int(len(frame)),
        "start_timestamp": frame["timestamp"].iloc[0].isoformat(),
        "end_timestamp": frame["timestamp"].iloc[-1].isoformat(),
        "positive_rate": float(target.mean()),
        "accuracy": float(accuracy_score(target, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "mean_up_probability": float(probabilities.mean()),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def train_and_evaluate(frame: pd.DataFrame, model: Any, model_name: str) -> tuple[Any, dict[str, Any]]:
    """Train a classifier and return the fitted model with evaluation metrics."""
    train_frame, validation_frame, test_frame = split_time_series(frame)
    model.fit(train_frame[FEATURE_COLUMNS], train_frame[TARGET_COLUMN])

    metrics = {
        "model_type": model_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "splits": {
            "train_rows": int(len(train_frame)),
            "validation_rows": int(len(validation_frame)),
            "test_rows": int(len(test_frame)),
        },
        "validation": evaluate_split(model, validation_frame, "validation"),
        "test": evaluate_split(model, test_frame, "test"),
    }
    return model, metrics


def persist_artifacts(
    model: Any,
    metrics: dict[str, Any],
    model_path: Path,
    metrics_path: Path,
) -> None:
    """Persist the trained model and metrics summary."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def parse_common_args(
    description: str,
    default_model_path: Path,
    default_metrics_path: Path,
) -> argparse.Namespace:
    """Parse standard CLI arguments for model-training entrypoints."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Processed feature CSV path. Defaults to {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=default_model_path,
        help=f"Model artifact path. Defaults to {default_model_path}.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=default_metrics_path,
        help=f"Metrics JSON path. Defaults to {default_metrics_path}.",
    )
    return parser.parse_args()
