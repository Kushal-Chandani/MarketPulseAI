"""Train and evaluate a dummy majority-class baseline."""

from __future__ import annotations

from pathlib import Path

from sklearn.dummy import DummyClassifier

from .common import load_feature_frame, parse_common_args, persist_artifacts, train_and_evaluate

DEFAULT_MODEL_PATH = Path("artifacts/model_dummy.joblib")
DEFAULT_METRICS_PATH = Path("artifacts/metrics_dummy.json")


def build_model() -> DummyClassifier:
    """Build the dummy baseline classifier."""
    return DummyClassifier(strategy="prior")


def main() -> None:
    """Run dummy baseline training."""
    args = parse_common_args(
        description="Train and evaluate a dummy baseline model on XAUUSD features.",
        default_model_path=DEFAULT_MODEL_PATH,
        default_metrics_path=DEFAULT_METRICS_PATH,
    )
    frame = load_feature_frame(args.input)
    model, metrics = train_and_evaluate(frame, build_model(), "dummy_prior")
    persist_artifacts(model, metrics, args.model_output, args.metrics_output)
    print(
        "Saved trained model to "
        f"{args.model_output} and evaluation metrics to {args.metrics_output}."
    )


if __name__ == "__main__":
    main()
