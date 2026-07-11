"""Train and evaluate a histogram gradient boosting classifier."""

from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier

from .common import load_feature_frame, parse_common_args, persist_artifacts, train_and_evaluate

DEFAULT_MODEL_PATH = Path("artifacts/model_hist_gradient_boosting.joblib")
DEFAULT_METRICS_PATH = Path("artifacts/metrics_hist_gradient_boosting.json")


def build_model() -> HistGradientBoostingClassifier:
    """Build the histogram gradient boosting classifier."""
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=3,
        max_iter=200,
        min_samples_leaf=25,
        random_state=42,
    )


def main() -> None:
    """Run histogram gradient boosting training."""
    args = parse_common_args(
        description="Train and evaluate a histogram gradient boosting model on XAUUSD features.",
        default_model_path=DEFAULT_MODEL_PATH,
        default_metrics_path=DEFAULT_METRICS_PATH,
    )
    frame = load_feature_frame(args.input)
    model, metrics = train_and_evaluate(
        frame,
        build_model(),
        "hist_gradient_boosting",
    )
    persist_artifacts(model, metrics, args.model_output, args.metrics_output)
    print(
        "Saved trained model to "
        f"{args.model_output} and evaluation metrics to {args.metrics_output}."
    )


if __name__ == "__main__":
    main()
