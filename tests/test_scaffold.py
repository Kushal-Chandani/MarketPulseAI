"""Smoke tests for the initial project scaffold."""

from marketpulseai import __version__
from marketpulseai.api.main import app


def test_package_version_exposed() -> None:
    assert __version__ == "0.1.0"


def test_fastapi_app_metadata() -> None:
    assert app.title == "MarketPulseAI API"
