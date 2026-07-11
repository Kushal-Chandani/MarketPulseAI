# MarketPulseAI

MarketPulseAI is a starter repository for an hourly `XAUUSD` next-candle direction classification project.

## Goal

Build a reproducible machine learning pipeline that:

- downloads historical hourly `XAUUSD` OHLC data from a free source
- engineers point-in-time-safe predictive features
- trains a simple classifier for the next candle's direction
- serves the latest prediction through a FastAPI endpoint

## Scope

This repository will be built in stages:

1. Project scaffold
2. Data ingestion
3. Feature engineering
4. Model training and evaluation
5. FastAPI inference service

## Current Status

Scaffold only.

The codebase currently includes:

- a `src`-layout Python package
- placeholder CLI entrypoints for data fetching and model training
- a placeholder FastAPI application
- test scaffolding and artifact directories

## Planned Entry Points

```bash
python -m marketpulseai.data.fetch
python -m marketpulseai.model.train
uvicorn marketpulseai.api.main:app --reload
```
