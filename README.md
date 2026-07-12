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

Inference API is implemented.

The codebase currently includes:

- a `src`-layout Python package
- a working hourly `XAU/USD` fetch command
- a working feature engineering command for next-candle classification
- a working model training and evaluation command
- a FastAPI application serving the current best model
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

## Training And Evaluation

The training step uses:

- a chronological split: 70% train, 15% validation, 15% test
- a `StandardScaler` + logistic regression pipeline
- held-out validation and test metrics saved to `artifacts/metrics.json`

The trained model artifact is saved to `artifacts/model.joblib`.

Additional model experiment entrypoints are available for comparison:

- `python -m marketpulseai.model.train_dummy`
- `python -m marketpulseai.model.train_logistic`
- `python -m marketpulseai.model.train_random_forest`
- `python -m marketpulseai.model.train_hist_gradient_boosting`

Each writes its own model and metrics files under `artifacts/`.

### Model Comparison

The current best model is `random_forest`, selected because it has the strongest held-out validation and test balanced accuracy among the models tried so far.

| Model | Validation Accuracy | Validation Balanced Acc. | Test Accuracy | Test Balanced Acc. |
| --- | ---: | ---: | ---: | ---: |
| Dummy prior | 0.5060 | 0.5000 | 0.5160 | 0.5000 |
| Logistic regression | 0.5087 | 0.5030 | 0.5133 | 0.4984 |
| Random forest | 0.5768 | 0.5745 | 0.5600 | 0.5535 |
| Hist. gradient boosting | 0.5794 | 0.5777 | 0.5467 | 0.5430 |

## API

The API currently serves the `random_forest` artifact by default:

- model artifact: `artifacts/model_random_forest.joblib`
- metrics artifact: `artifacts/metrics_random_forest.json`
- feature input: `data/processed/xauusd_1h_features.csv`

Available endpoints:

- `GET /health`
- `GET /model/info`
- `GET /predict/latest`
