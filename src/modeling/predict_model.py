"""Generate current forward-risk predictions from the trained classifier."""

from __future__ import annotations

import argparse
import pickle

import pandas as pd

from src.utils.config import ProjectConfig, load_config
from src.utils.exceptions import DataValidationError, MissingInputError
from src.utils.io import read_parquet, write_parquet
from src.utils.paths import get_paths


def _collapse_merge_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Coalesce duplicated merge columns like foo_x / foo_y into foo."""
    frame = frame.copy()
    duplicate_bases = sorted(
        {
            column[:-2]
            for column in frame.columns
            if column.endswith("_x") and f"{column[:-2]}_y" in frame.columns
        }
    )

    for base in duplicate_bases:
        left = frame[f"{base}_x"]
        right = frame[f"{base}_y"]
        frame[base] = left.combine_first(right)

    columns_to_drop = [
        column
        for base in duplicate_bases
        for column in (f"{base}_x", f"{base}_y")
        if column in frame.columns
    ]
    if columns_to_drop:
        frame = frame.drop(columns=columns_to_drop)

    return frame


def run(config: ProjectConfig | None = None) -> pd.DataFrame:
    """Predict 12-month risk-increase probabilities for the latest available year."""
    config = config or load_config()
    paths = get_paths()

    if not paths.model_path.exists():
        raise MissingInputError(
            "Trained model artifact is missing. Run `python -m src.cli train-model` first."
        )

    with paths.model_path.open("rb") as handle:
        artifact = pickle.load(handle)

    feature_store = read_parquet(
        paths.feature_store,
        "Feature store is missing. Run `python -m src.cli build-feature-store` first.",
    )
    risk_scores = read_parquet(
        paths.risk_scores,
        "Risk scores are missing. Run `python -m src.cli score-risk` first.",
    )

    scoring_frame = feature_store.merge(risk_scores, on=["hex_id", "year"], how="left")
    scoring_frame = _collapse_merge_duplicates(scoring_frame)

    if scoring_frame.empty:
        raise MissingInputError("No scored feature rows are available for prediction.")

    latest_year = int(scoring_frame["year"].max())
    current = scoring_frame[scoring_frame["year"] == latest_year].copy()

    if "risk_score_t" not in current.columns and "risk_score" in current.columns:
        current["risk_score_t"] = current["risk_score"]

    feature_columns = artifact["feature_columns"]
    missing_columns = [column for column in feature_columns if column not in current.columns]
    if missing_columns:
        raise DataValidationError(
            "Prediction input is missing model feature columns: "
            f"{missing_columns}. Rebuild the feature store / risk scores or retrain the model."
        )

    X = current[feature_columns]
    current["prob_risk_increase_12m"] = artifact["model"].predict_proba(X)[:, 1]

    predictions = current[["hex_id"]].copy()
    predictions["prediction_year"] = latest_year + 1
    predictions["prob_risk_increase_12m"] = current["prob_risk_increase_12m"]

    write_parquet(predictions, paths.predictions_current)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict forward risk increase probabilities.")
    parser.add_argument("--config", default=None, help="Path to the YAML config file.")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()