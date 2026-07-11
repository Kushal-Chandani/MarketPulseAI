"""Train and evaluate a random forest classifier."""

from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from .common import load_feature_frame, parse_common_args, persist_artifacts, train_and_evaluate

DEFAULT_MODEL_PATH = Path("artifacts/model_random_forest.joblib")
DEFAULT_METRICS_PATH = Path("artifacts/metrics_random_forest.json")


def build_model() -> RandomForestClassifier:
    """Build the random forest classifier."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )


def main() -> None:
    """Run random forest training."""
    args = parse_common_args(
        description="Train and evaluate a random forest model on XAUUSD features.",
        default_model_path=DEFAULT_MODEL_PATH,
        default_metrics_path=DEFAULT_METRICS_PATH,
    )
    frame = load_feature_frame(args.input)
    model, metrics = train_and_evaluate(frame, build_model(), "random_forest")
    persist_artifacts(model, metrics, args.model_output, args.metrics_output)
    print(
        "Saved trained model to "
        f"{args.model_output} and evaluation metrics to {args.metrics_output}."
    )


if __name__ == "__main__":
    main()
