"""Train and evaluate a logistic regression classifier."""

from __future__ import annotations

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import load_feature_frame, parse_common_args, persist_artifacts, train_and_evaluate

DEFAULT_MODEL_PATH = Path("artifacts/model_logistic.joblib")
DEFAULT_METRICS_PATH = Path("artifacts/metrics_logistic.json")


def build_model() -> Pipeline:
    """Build the logistic regression pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def main() -> None:
    """Run logistic regression training."""
    args = parse_common_args(
        description="Train and evaluate a logistic regression model on XAUUSD features.",
        default_model_path=DEFAULT_MODEL_PATH,
        default_metrics_path=DEFAULT_METRICS_PATH,
    )
    frame = load_feature_frame(args.input)
    model, metrics = train_and_evaluate(frame, build_model(), "logistic_regression")
    persist_artifacts(model, metrics, args.model_output, args.metrics_output)
    print(
        "Saved trained model to "
        f"{args.model_output} and evaluation metrics to {args.metrics_output}."
    )


if __name__ == "__main__":
    main()
