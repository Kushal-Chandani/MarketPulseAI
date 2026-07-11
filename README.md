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

Data ingestion is implemented.

The codebase currently includes:

- a `src`-layout Python package
- a working hourly `XAU/USD` fetch command
- a working feature engineering command for next-candle classification
- a placeholder CLI entrypoint for model training
- a placeholder FastAPI application
- test scaffolding and artifact directories

## Planned Entry Points

```bash
python -m marketpulseai.data.fetch
python -m marketpulseai.features.build
python -m marketpulseai.model.train
uvicorn marketpulseai.api.main:app --reload
```

These commands work when run from the repository root. The repo includes a small top-level package shim so the `src` layout is importable locally without requiring an editable install first.

## Data Source

The ingestion step uses Twelve Data's time series endpoint with:

- `symbol=XAU/USD`
- `interval=1h`

You need a free Twelve Data API key in `TWELVEDATA_API_KEY` before running the fetch command.

You can provide it either:

- in your shell environment, or
- in a local `.env` file at the repo root

Example `.env`:

```env
TWELVEDATA_API_KEY=your_api_key_here
```

## Engineered Features

The feature step produces a processed dataset with five predictors and one binary target:

- `log_return_1`: one-candle log return
- `momentum_3`: three-candle percentage momentum
- `range_pct`: intrabar high-low range normalized by close
- `volatility_6`: six-candle rolling standard deviation of log returns
- `ema_gap_8`: distance from an 8-period exponential moving average
- `target_up`: `1` when the next candle closes above the current close, else `0`
