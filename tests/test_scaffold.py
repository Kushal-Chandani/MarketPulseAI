"""Smoke tests for the initial project scaffold."""

from marketpulseai import __version__
from marketpulseai.api.main import app
from marketpulseai.data.fetch import build_params


def test_package_version_exposed() -> None:
    assert __version__ == "0.1.0"


def test_fastapi_app_metadata() -> None:
    assert app.title == "MarketPulseAI API"


def test_twelvedata_params_include_hourly_xauusd() -> None:
    params = build_params("demo-key")
    assert params["symbol"] == "XAU/USD"
    assert params["interval"] == "1h"
    assert params["outputsize"] == "5000"
