"""Tests for the Twelve Data ingestion flow."""

from __future__ import annotations

import pytest

from marketpulseai.data.fetch import build_metadata, parse_twelvedata_payload


def test_parse_twelvedata_payload_returns_sorted_frame() -> None:
    payload = {
        "meta": {
            "symbol": "XAU/USD",
            "interval": "1h",
            "exchange_timezone": "UTC",
        },
        "values": [
            {
                "datetime": "2026-07-11 23:00:00",
                "open": "3331.10",
                "high": "3335.40",
                "low": "3329.20",
                "close": "3334.80",
            },
            {
                "datetime": "2026-07-11 22:00:00",
                "open": "3327.00",
                "high": "3332.10",
                "low": "3325.90",
                "close": "3331.10",
            },
        ],
    }

    frame = parse_twelvedata_payload(payload)

    assert frame["timestamp"].is_monotonic_increasing
    assert frame.iloc[0]["close"] == pytest.approx(3331.10)
    assert frame.iloc[1]["close"] == pytest.approx(3334.80)


def test_build_metadata_summarizes_frame() -> None:
    payload = {
        "meta": {
            "symbol": "XAU/USD",
            "interval": "1h",
            "exchange_timezone": "UTC",
        },
        "values": [
            {
                "datetime": "2026-07-11 23:00:00",
                "open": "3331.10",
                "high": "3335.40",
                "low": "3329.20",
                "close": "3334.80",
            }
        ],
    }

    frame = parse_twelvedata_payload(payload)
    metadata = build_metadata(payload, frame)

    assert metadata["source"] == "twelvedata"
    assert metadata["symbol"] == "XAUUSD"
    assert metadata["rows"] == 1
    assert metadata["interval"] == "1h"


def test_parse_twelvedata_payload_raises_on_provider_error() -> None:
    payload = {"status": "error", "message": "Invalid API key."}

    with pytest.raises(RuntimeError, match="Twelve Data error"):
        parse_twelvedata_payload(payload)
