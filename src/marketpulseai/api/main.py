"""FastAPI application for latest XAUUSD direction inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from marketpulseai.features.build import FEATURE_COLUMNS

DEFAULT_MODEL_PATH = Path("artifacts/model_random_forest.joblib")
DEFAULT_METRICS_PATH = Path("artifacts/metrics_random_forest.json")
DEFAULT_FEATURE_DATA_PATH = Path("data/processed/xauusd_1h_features.csv")

app = FastAPI(
    title="MarketPulseAI API",
    description="FastAPI service for XAUUSD next-candle direction inference.",
    version="0.1.0",
)


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> Any:
    """Load the configured model artifact."""
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model artifact not found at {model_path}. Train the model first.",
        )
    return joblib.load(model_path)


def load_metrics(metrics_path: Path = DEFAULT_METRICS_PATH) -> dict[str, Any]:
    """Load the configured metrics artifact."""
    if not metrics_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Metrics artifact not found at {metrics_path}. Train the model first.",
        )
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def load_latest_feature_row(
    feature_data_path: Path = DEFAULT_FEATURE_DATA_PATH,
) -> pd.Series:
    """Load the most recent engineered feature row."""
    if not feature_data_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Feature dataset not found at {feature_data_path}. Build features first.",
        )

    frame = pd.read_csv(feature_data_path, parse_dates=["timestamp"])
    if frame.empty:
        raise HTTPException(status_code=503, detail="Feature dataset is empty.")

    missing_columns = [column for column in ["timestamp", *FEATURE_COLUMNS] if column not in frame.columns]
    if missing_columns:
        raise HTTPException(
            status_code=503,
            detail=f"Feature dataset is missing required columns: {missing_columns}",
        )

    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame.iloc[-1]


def build_prediction_payload(
    model: Any,
    latest_row: pd.Series,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build the API response payload for the latest prediction."""
    feature_frame = pd.DataFrame([latest_row[FEATURE_COLUMNS].to_dict()], columns=FEATURE_COLUMNS)
    predicted_class = int(model.predict(feature_frame)[0])
    probabilities = model.predict_proba(feature_frame)[0]
    model_classes = list(getattr(model, "classes_", []))

    if 1 in model_classes:
        up_probability = float(probabilities[model_classes.index(1)])
    else:
        up_probability = 1.0 if predicted_class == 1 else 0.0

    return {
        "symbol": "XAUUSD",
        "timeframe": "1h",
        "model_type": metrics["model_type"],
        "as_of_timestamp": pd.Timestamp(latest_row["timestamp"]).isoformat(),
        "predicted_direction": "up" if predicted_class == 1 else "down",
        "up_probability": up_probability,
        "features": {
            column: float(latest_row[column])
            for column in FEATURE_COLUMNS
        },
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Return service health and artifact availability."""
    return {
        "status": "ok",
        "model_ready": DEFAULT_MODEL_PATH.exists(),
        "metrics_ready": DEFAULT_METRICS_PATH.exists(),
        "features_ready": DEFAULT_FEATURE_DATA_PATH.exists(),
        "selected_model": "random_forest",
    }


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    """Return metadata about the currently served model."""
    metrics = load_metrics()
    return {
        "symbol": "XAUUSD",
        "timeframe": "1h",
        "served_model_artifact": str(DEFAULT_MODEL_PATH),
        "served_metrics_artifact": str(DEFAULT_METRICS_PATH),
        "model_type": metrics["model_type"],
        "feature_columns": metrics["feature_columns"],
        "target_column": metrics["target_column"],
        "validation": metrics["validation"],
        "test": metrics["test"],
    }


@app.get("/predict/latest")
def predict_latest() -> dict[str, Any]:
    """Serve the latest prediction using the selected best model."""
    model = load_model()
    metrics = load_metrics()
    latest_row = load_latest_feature_row()
    return build_prediction_payload(model, latest_row, metrics)
