# MarketPulseAI

MarketPulseAI is a machine learning project that predicts whether the next hourly `XAUUSD` candle will close up or down.

In simple terms:

- `XAUUSD` is the price of gold quoted in US dollars
- the project looks at recent hourly gold price behavior
- it creates a few simple signals from that price history
- it uses those signals to predict the direction of the next hour
- it exposes the latest prediction through a FastAPI API

This repository is designed as a clean, reviewable take-home style project. The main focus is not flashy model complexity. The focus is disciplined data handling, no look-ahead leakage, honest time-based evaluation, and a usable inference API.

## What This Project Does

If a non-technical person were using this project, the summary would be:

1. Download recent hourly gold price data
2. Turn that data into a few meaningful indicators
3. Train models to guess whether the next hour closes higher or lower
4. Compare the models fairly on future data they have not seen before
5. Serve the best model through an API so the latest prediction can be requested on demand

This project predicts **direction**, not exact price.

That means the output is:

- `up`: the next hourly candle is expected to close above the current close
- `down`: the next hourly candle is expected to close at or below the current close

## Why This Project Exists

Financial machine learning projects often look better than they really are because of weak evaluation or data leakage.

This project was built to avoid that.

Key rules followed here:

- all features use only information available at the time of prediction
- the target is the **next** candle's direction, not the current candle
- the dataset is split by time, not randomly
- evaluation is reported on a true holdout period
- weaker baselines are included so model gains are visible and honest

## Project Flow

The full workflow is:

```text
Raw hourly XAUUSD candles
    -> feature engineering
    -> train / validation / test split by time
    -> multiple classifier experiments
    -> select the best held-out model
    -> serve latest prediction with FastAPI
```

## Data Source

Historical hourly candles are downloaded from Twelve Data.

- Symbol: `XAU/USD`
- Interval: `1h`
- Output used in this project: hourly OHLC candles

You need a Twelve Data API key in a local `.env` file or in your shell environment.

Example:

```env
TWELVEDATA_API_KEY=your_api_key_here
```

## Features Used

The current version uses five hand-engineered features:

1. `log_return_1`
Recent one-candle log return.

2. `momentum_3`
Three-candle momentum, which captures short-term trend.

3. `range_pct`
The candle's high-low range divided by close, which measures how large the move was inside the hour.

4. `volatility_6`
Rolling standard deviation of recent log returns over six candles.

5. `ema_gap_8`
Distance between the current close and an 8-period exponential moving average.

These were chosen because they are simple, interpretable, and reasonable for a baseline directional model.

## Target Definition

The target is:

- `1` if `close[t+1] > close[t]`
- `0` otherwise

So the model never sees the future target when generating the input features for row `t`.

## Train / Validation / Test Split

The data is split chronologically:

- 70% train
- 15% validation
- 15% test

This matters because random shuffling would create an unrealistic evaluation for time series data. In real usage, a model only has access to the past, so the test set must come after the training period.

Current split sizes from the generated dataset:

- train: `3495` rows
- validation: `749` rows
- test: `750` rows

## Models Tried

The following models were trained and evaluated:

- Dummy prior baseline
- Logistic regression
- Random forest
- Histogram gradient boosting

The dummy baseline is important because it shows what happens if the model simply predicts the majority class. If a more advanced model cannot beat that baseline meaningfully, it is not useful.

## Results

### Model Comparison

| Model | Validation Accuracy | Validation Balanced Acc. | Test Accuracy | Test Balanced Acc. |
| --- | ---: | ---: | ---: | ---: |
| Dummy prior | 0.5060 | 0.5000 | 0.5160 | 0.5000 |
| Logistic regression | 0.5087 | 0.5030 | 0.5133 | 0.4984 |
| Random forest | 0.5768 | 0.5745 | 0.5600 | 0.5535 |
| Hist. gradient boosting | 0.5794 | 0.5777 | 0.5467 | 0.5430 |

### Selected Model

The API currently serves the `random_forest` model.

Why:

- it clearly beats the dummy baseline
- it outperforms logistic regression on both validation and test
- it has the strongest held-out test balanced accuracy among the tested models
- it is a reasonable non-linear model for a small tabular feature set

### Honest Interpretation

The results are better than chance, but not dramatically so.

That is a realistic outcome for short-horizon financial direction prediction. A model with about `0.56` test accuracy and `0.5535` balanced accuracy is not a magic trading system. It is simply the strongest model in this baseline experiment.

This is exactly why the evaluation is useful: it shows modest predictive signal without pretending the model is stronger than it is.

## API

The FastAPI service serves the latest prediction from the best current model.

Current default artifacts:

- model: `artifacts/model_random_forest.joblib`
- metrics: `artifacts/metrics_random_forest.json`
- features input: `data/processed/xauusd_1h_features.csv`

Available endpoints:

- `GET /health`
- `GET /model/info`
- `GET /predict/latest`

### Example Response

`GET /predict/latest` returns a payload like:

```json
{
  "symbol": "XAUUSD",
  "timeframe": "1h",
  "model_type": "random_forest",
  "as_of_timestamp": "2026-07-12T06:00:00",
  "predicted_direction": "down",
  "up_probability": 0.2482,
  "features": {
    "log_return_1": -0.000024,
    "momentum_3": -0.000018,
    "range_pct": 0.000044,
    "volatility_6": 0.000011,
    "ema_gap_8": -0.000010
  }
}
```

## Repository Structure

```text
src/marketpulseai/data/        data download
src/marketpulseai/features/    feature engineering
src/marketpulseai/model/       training and evaluation
src/marketpulseai/api/         FastAPI inference service
data/raw/                      downloaded candles
data/processed/                engineered feature dataset
artifacts/                     trained models and metrics
tests/                         automated tests
```

## How To Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your API key

Create a `.env` file in the repo root:

```env
TWELVEDATA_API_KEY=your_api_key_here
```

### 3. Download raw data

```bash
python -m marketpulseai.data.fetch
```

### 4. Build features

```bash
python -m marketpulseai.features.build
```

### 5. Train the default model

```bash
python -m marketpulseai.model.train
```

### 6. Train comparison models

```bash
python -m marketpulseai.model.train_dummy
python -m marketpulseai.model.train_logistic
python -m marketpulseai.model.train_random_forest
python -m marketpulseai.model.train_hist_gradient_boosting
```

### 7. Run the API

```bash
uvicorn marketpulseai.api.main:app --reload
```

### 8. Call the endpoints

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model/info
curl http://127.0.0.1:8000/predict/latest
```

## Testing

Run the test suite with:

```bash
pytest
```

The test coverage includes:

- data ingestion parsing and validation
- feature engineering alignment
- target construction
- chronological train/validation/test splitting
- model training metric structure
- API helper and endpoint behavior

## Important Limitations

This project is intentionally honest about its limits:

- it uses a small set of handcrafted price-only features
- it does not include transaction costs, slippage, or trading rules
- it predicts next-candle direction, not profitability
- it is trained on one instrument and one timeframe only
- model performance is modest and should not be overstated

## Future Improvements

Possible next steps:

- richer technical features
- rolling or walk-forward retraining
- probability threshold tuning on validation data
- calibration checks for predicted probabilities
- a trading simulation layer with costs and execution assumptions
- feature importance and model monitoring outputs

## Summary

MarketPulseAI is a clean baseline machine learning system for hourly gold direction prediction.

Its value is not that it "solves" the market. Its value is that it demonstrates a disciplined workflow:

- clean data ingestion
- no look-ahead leakage
- time-aware evaluation
- multiple model comparisons
- transparent reporting
- simple API-based serving
