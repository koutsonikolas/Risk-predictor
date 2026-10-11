"""Build the supervised training table for 12-month risk increase prediction."""

from __future__ import annotations

import argparse

import pandas as pd

from src.utils.config import ProjectConfig, load_config
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
    """Create the yearly modeling table with 12-month forward targets."""
    config = config or load_config()
    paths = get_paths()

    features = read_parquet(
        paths.feature_store,
        "Feature store is missing. Run `python -m src.cli build-feature-store` first.",
    )
    scores = read_parquet(
        paths.risk_scores,
        "Risk scores are missing. Run `python -m src.cli score-risk` first.",
    )

    current = features.merge(scores, on=["hex_id", "year"], how="left")
    current = _collapse_merge_duplicates(current)

    future_scores = scores[["hex_id", "year", "risk_score"]].rename(
        columns={"year": "prediction_year", "risk_score": "risk_score_t_plus_1"}
    )

    current["prediction_year"] = current["year"] + 1
    training = current.merge(
        future_scores,
        on=["hex_id", "prediction_year"],
        how="left",
    )

    training = training.rename(columns={"risk_score": "risk_score_t"})
    training["risk_delta_12m"] = training["risk_score_t_plus_1"] - training["risk_score_t"]
    training["target_increase_12m"] = (training["risk_delta_12m"] >= 8.0).astype("Int64")
    training = training[training["risk_score_t_plus_1"].notna()].reset_index(drop=True)

    ordered_front = [
        "hex_id",
        "year",
        "prediction_year",
        "risk_score_t",
        "risk_score_t_plus_1",
        "risk_delta_12m",
        "target_increase_12m",
    ]
    trailing = [column for column in training.columns if column not in ordered_front]
    training = training[ordered_front + trailing]

    write_parquet(training, paths.training_table)
    return training


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 12-month forward training table.")
    parser.add_argument("--config", default=None, help="Path to the YAML config file.")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()