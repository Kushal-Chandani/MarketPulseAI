"""Tests for model training and chronological evaluation."""

from __future__ import annotations

import pandas as pd

from marketpulseai.features.build import FEATURE_COLUMNS, TARGET_COLUMN
from marketpulseai.model.train import split_time_series, train_and_evaluate


def make_feature_frame(rows: int = 40) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=rows, freq="h")
    records: list[dict[str, float | int | pd.Timestamp]] = []

    for index, timestamp in enumerate(timestamps):
        base = float(index + 1)
        records.append(
            {
                "timestamp": timestamp,
                "open": 1000.0 + base,
                "high": 1001.0 + base,
                "low": 999.0 + base,
                "close": 1000.5 + base,
                "log_return_1": 0.001 * ((index % 5) + 1),
                "momentum_3": 0.002 * ((index % 3) - 1),
                "range_pct": 0.003 + (index * 0.00001),
                "volatility_6": 0.004 + ((index % 4) * 0.0001),
                "ema_gap_8": -0.002 + ((index % 6) * 0.0002),
                "target_up": int(index % 2 == 0),
            }
        )

    return pd.DataFrame.from_records(records)


def test_split_time_series_preserves_order() -> None:
    frame = make_feature_frame(rows=20)
    train_frame, validation_frame, test_frame = split_time_series(frame)

    assert len(train_frame) == 14
    assert len(validation_frame) == 3
    assert len(test_frame) == 3
    assert train_frame["timestamp"].max() < validation_frame["timestamp"].min()
    assert validation_frame["timestamp"].max() < test_frame["timestamp"].min()


def test_train_and_evaluate_returns_expected_metrics_shape() -> None:
    model, metrics = train_and_evaluate(make_feature_frame())

    assert list(metrics["feature_columns"]) == FEATURE_COLUMNS
    assert metrics["target_column"] == TARGET_COLUMN
    assert metrics["model_type"] == "logistic_regression"
    assert metrics["splits"]["train_rows"] > metrics["splits"]["validation_rows"] > 0
    assert metrics["splits"]["test_rows"] > 0

    for split_name in ["validation", "test"]:
        split_metrics = metrics[split_name]
        assert split_metrics["split"] == split_name
        assert 0.0 <= split_metrics["accuracy"] <= 1.0
        assert 0.0 <= split_metrics["balanced_accuracy"] <= 1.0
        assert 0.0 <= split_metrics["precision"] <= 1.0
        assert 0.0 <= split_metrics["recall"] <= 1.0
        assert 0.0 <= split_metrics["f1"] <= 1.0
        assert 0.0 <= split_metrics["mean_up_probability"] <= 1.0
        assert set(split_metrics["confusion_matrix"]) == {"tn", "fp", "fn", "tp"}

    assert hasattr(model, "predict")
