"""Tests for the inference API helpers and endpoints."""

from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from marketpulseai.api.main import (
    app,
    build_prediction_payload,
    load_latest_feature_row,
    load_metrics,
    load_model,
)
from marketpulseai.features.build import FEATURE_COLUMNS


TEST_SCRATCH_ROOT = Path("tests/_scratch_api")


def make_feature_dataset(path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-07-12 05:00:00",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "log_return_1": 0.01,
                "momentum_3": 0.02,
                "range_pct": 0.03,
                "volatility_6": 0.04,
                "ema_gap_8": 0.05,
                "target_up": 1,
            },
            {
                "timestamp": "2026-07-12 06:00:00",
                "open": 101.0,
                "high": 102.0,
                "low": 100.0,
                "close": 101.5,
                "log_return_1": 0.011,
                "momentum_3": 0.021,
                "range_pct": 0.031,
                "volatility_6": 0.041,
                "ema_gap_8": 0.051,
                "target_up": 0,
            },
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def make_dummy_model(path: Path) -> None:
    model = DummyClassifier(strategy="constant", constant=1)
    training_frame = pd.DataFrame(
        [
            {column: 0.1 for column in FEATURE_COLUMNS},
            {column: 0.2 for column in FEATURE_COLUMNS},
        ]
    )
    model.fit(training_frame, [1, 1])
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def make_metrics_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
{
  "model_type": "dummy_prior",
  "feature_columns": ["log_return_1", "momentum_3", "range_pct", "volatility_6", "ema_gap_8"],
  "target_column": "target_up",
  "validation": {"accuracy": 0.5},
  "test": {"accuracy": 0.5}
}
        """.strip(),
        encoding="utf-8",
    )


def make_scratch_dir() -> Path:
    scratch_dir = TEST_SCRATCH_ROOT / uuid.uuid4().hex
    scratch_dir.mkdir(parents=True, exist_ok=False)
    return scratch_dir


def test_load_latest_feature_row_returns_most_recent() -> None:
    scratch_dir = make_scratch_dir()
    feature_path = scratch_dir / "features.csv"
    make_feature_dataset(feature_path)

    try:
        latest_row = load_latest_feature_row(feature_path)
        assert pd.Timestamp(latest_row["timestamp"]).isoformat() == "2026-07-12T06:00:00"
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def test_build_prediction_payload_uses_model_outputs() -> None:
    scratch_dir = make_scratch_dir()
    feature_path = scratch_dir / "features.csv"
    model_path = scratch_dir / "model.joblib"
    metrics_path = scratch_dir / "metrics.json"
    make_feature_dataset(feature_path)
    make_dummy_model(model_path)
    make_metrics_file(metrics_path)

    try:
        payload = build_prediction_payload(
            load_model(model_path),
            load_latest_feature_row(feature_path),
            load_metrics(metrics_path),
        )

        assert payload["predicted_direction"] == "up"
        assert payload["model_type"] == "dummy_prior"
        assert set(payload["features"]) == set(FEATURE_COLUMNS)
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def test_health_endpoint_reports_status() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
