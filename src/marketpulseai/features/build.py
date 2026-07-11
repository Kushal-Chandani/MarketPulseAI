"""Feature engineering for hourly XAUUSD direction modeling."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT_PATH = Path("data/raw/xauusd_1h.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/xauusd_1h_features.csv")

FEATURE_COLUMNS = [
    "log_return_1",
    "momentum_3",
    "range_pct",
    "volatility_6",
    "ema_gap_8",
]

TARGET_COLUMN = "target_up"


def load_raw_ohlc(input_path: Path) -> pd.DataFrame:
    """Load raw OHLC data from disk."""
    frame = pd.read_csv(input_path, parse_dates=["timestamp"])
    validate_raw_ohlc(frame)
    return frame.sort_values("timestamp").reset_index(drop=True)


def validate_raw_ohlc(frame: pd.DataFrame) -> None:
    """Validate the raw OHLC frame before feature engineering."""
    expected_columns = ["timestamp", "open", "high", "low", "close"]
    if list(frame.columns) != expected_columns:
        raise RuntimeError(f"Unexpected raw columns: {frame.columns.tolist()}")
    if frame.empty:
        raise RuntimeError("Raw OHLC frame is empty.")
    if frame["timestamp"].duplicated().any():
        raise RuntimeError("Raw OHLC frame contains duplicate timestamps.")


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create point-in-time-safe features and the next-candle target."""
    features = frame.copy()

    features["log_return_1"] = np.log(features["close"] / features["close"].shift(1))
    features["momentum_3"] = features["close"].pct_change(periods=3)
    features["range_pct"] = (features["high"] - features["low"]) / features["close"]
    features["volatility_6"] = features["log_return_1"].rolling(window=6).std()
    ema_8 = features["close"].ewm(span=8, adjust=False).mean()
    features["ema_gap_8"] = (features["close"] / ema_8) - 1.0
    features["target_up"] = (features["close"].shift(-1) > features["close"]).astype(
        "Int64"
    )

    # Drop warm-up rows for rolling features and the final row without a target.
    features = features.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(
        drop=True
    )
    features[TARGET_COLUMN] = features[TARGET_COLUMN].astype(int)
    return features


def persist_feature_frame(frame: pd.DataFrame, output_path: Path) -> None:
    """Write engineered features to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the feature builder."""
    parser = argparse.ArgumentParser(
        description="Build leakage-safe features for hourly XAUUSD direction modeling."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Raw OHLC CSV path. Defaults to {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Processed feature CSV path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the feature engineering entrypoint."""
    args = parse_args()
    raw_frame = load_raw_ohlc(args.input)
    feature_frame = engineer_features(raw_frame)
    persist_feature_frame(feature_frame, args.output)
    print(f"Saved {len(feature_frame)} engineered rows to {args.output}.")


if __name__ == "__main__":
    main()
