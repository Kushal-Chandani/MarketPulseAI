"""Placeholder FastAPI application."""

from fastapi import FastAPI

app = FastAPI(
    title="MarketPulseAI API",
    description="Placeholder API for XAUUSD next-candle direction inference.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return a basic service health response."""
    return {"status": "ok", "stage": "scaffold"}
