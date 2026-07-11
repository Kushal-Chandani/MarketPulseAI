"""Tests for feature engineering and target alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketpulseai.features.build import FEATURE_COLUMNS, TARGET_COLUMN, engineer_features


def make_raw_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=10, freq="h")
    close = pd.Series([100.0, 101.0, 103.0, 102.0, 104.0, 107.0, 106.0, 108.0, 109.0, 111.0])
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        }
    )


def test_engineer_features_creates_expected_columns() -> None:
    feature_frame = engineer_features(make_raw_frame())

    for column in ["timestamp", "open", "high", "low", "close", *FEATURE_COLUMNS, TARGET_COLUMN]:
        assert column in feature_frame.columns

    assert not feature_frame[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().any().any()


def test_target_uses_next_candle_close() -> None:
    raw_frame = make_raw_frame()
    feature_frame = engineer_features(raw_frame)

    first_timestamp = feature_frame.iloc[0]["timestamp"]
    raw_index = raw_frame.index[raw_frame["timestamp"] == first_timestamp][0]

    expected_target = int(raw_frame.loc[raw_index + 1, "close"] > raw_frame.loc[raw_index, "close"])
    assert feature_frame.iloc[0][TARGET_COLUMN] == expected_target


def test_rolling_features_match_expected_history_only() -> None:
    raw_frame = make_raw_frame()
    feature_frame = engineer_features(raw_frame)

    first_timestamp = feature_frame.iloc[0]["timestamp"]
    raw_index = raw_frame.index[raw_frame["timestamp"] == first_timestamp][0]

    expected_log_return = np.log(raw_frame.loc[raw_index, "close"] / raw_frame.loc[raw_index - 1, "close"])
    expected_momentum = (
        raw_frame.loc[raw_index, "close"] / raw_frame.loc[raw_index - 3, "close"]
    ) - 1.0
    expected_range_pct = (
        (raw_frame.loc[raw_index, "high"] - raw_frame.loc[raw_index, "low"])
        / raw_frame.loc[raw_index, "close"]
    )

    log_returns = np.log(raw_frame["close"] / raw_frame["close"].shift(1))
    trailing_returns = log_returns.iloc[raw_index - 5 : raw_index + 1]
    expected_volatility = trailing_returns.std(ddof=1)
    expected_ema = raw_frame.loc[:raw_index, "close"].ewm(span=8, adjust=False).mean().iloc[-1]
    expected_ema_gap = (raw_frame.loc[raw_index, "close"] / expected_ema) - 1.0

    row = feature_frame.iloc[0]
    assert row["log_return_1"] == pytest.approx(expected_log_return)
    assert row["momentum_3"] == pytest.approx(expected_momentum)
    assert row["range_pct"] == pytest.approx(expected_range_pct)
    assert row["volatility_6"] == pytest.approx(expected_volatility)
    assert row["ema_gap_8"] == pytest.approx(expected_ema_gap)
