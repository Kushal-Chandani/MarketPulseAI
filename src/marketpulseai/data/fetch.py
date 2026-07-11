"""Fetch hourly XAUUSD OHLC data from Twelve Data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_URL = "https://api.twelvedata.com/time_series"
DEFAULT_OUTPUT_PATH = Path("data/raw/xauusd_1h.csv")
DEFAULT_METADATA_PATH = Path("data/raw/xauusd_1h_metadata.json")
API_KEY_ENV_VAR = "TWELVEDATA_API_KEY"


def load_api_key() -> str:
    """Load the Twelve Data API key from the environment."""
    load_dotenv()
    api_key = os.getenv(API_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set the {API_KEY_ENV_VAR} environment variable "
            "with your free Twelve Data API key, or place it in a local .env file."
        )
    return api_key


def build_params(api_key: str) -> dict[str, str]:
    """Build query parameters for the hourly XAUUSD request."""
    return {
        "symbol": "XAU/USD",
        "interval": "1h",
        "outputsize": "5000",
        "apikey": api_key,
    }


def fetch_ohlc_frame(session: requests.Session | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download and normalize hourly XAUUSD candles from Twelve Data."""
    api_key = load_api_key()
    client = session or requests.Session()
    response = client.get(BASE_URL, params=build_params(api_key), timeout=30)
    response.raise_for_status()

    payload = response.json()
    frame = parse_twelvedata_payload(payload)
    metadata = build_metadata(payload, frame)
    return frame, metadata


def parse_twelvedata_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse the Twelve Data time series payload into a clean frame."""
    if payload.get("status") == "error":
        message = payload.get("message", "Unknown Twelve Data error.")
        raise RuntimeError(f"Twelve Data error: {message}")

    series = payload.get("values")
    if not isinstance(series, list) or not series:
        raise RuntimeError("Unexpected Twelve Data response: missing 'values'.")

    rows: list[dict[str, Any]] = []
    for values in series:
        rows.append(
            {
                "timestamp": pd.to_datetime(values["datetime"], utc=False),
                "open": float(values["open"]),
                "high": float(values["high"]),
                "low": float(values["low"]),
                "close": float(values["close"]),
            }
        )

    frame = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    validate_frame(frame)
    return frame


def validate_frame(frame: pd.DataFrame) -> None:
    """Ensure the fetched frame is suitable for downstream use."""
    expected_columns = ["timestamp", "open", "high", "low", "close"]
    if list(frame.columns) != expected_columns:
        raise RuntimeError(f"Unexpected columns: {frame.columns.tolist()}")
    if frame.empty:
        raise RuntimeError("Fetched frame is empty.")
    if frame["timestamp"].duplicated().any():
        raise RuntimeError("Fetched frame contains duplicate timestamps.")
    if not frame["timestamp"].is_monotonic_increasing:
        raise RuntimeError("Fetched frame timestamps are not sorted ascending.")


def build_metadata(payload: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    """Build a small metadata payload for reproducibility."""
    meta = payload.get("meta", {})
    return {
        "source": "twelvedata",
        "symbol": "XAUUSD",
        "interval": meta.get("interval", "1h"),
        "timezone": meta.get("exchange_timezone") or meta.get("timezone"),
        "last_refreshed": frame["timestamp"].iloc[-1].isoformat(),
        "rows": int(len(frame)),
        "start_timestamp": frame["timestamp"].iloc[0].isoformat(),
        "end_timestamp": frame["timestamp"].iloc[-1].isoformat(),
    }


def persist_outputs(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    output_path: Path,
    metadata_path: Path,
) -> None:
    """Write the normalized OHLC data and metadata to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the fetch command."""
    parser = argparse.ArgumentParser(
        description="Fetch hourly XAUUSD OHLC data from Twelve Data."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"CSV output path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help=f"Metadata JSON output path. Defaults to {DEFAULT_METADATA_PATH}.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the data ingestion entrypoint."""
    args = parse_args()
    frame, metadata = fetch_ohlc_frame()
    persist_outputs(frame, metadata, args.output, args.metadata_output)
    print(
        f"Saved {len(frame)} hourly XAUUSD rows to {args.output} "
        f"and metadata to {args.metadata_output}."
    )


if __name__ == "__main__":
    main()
