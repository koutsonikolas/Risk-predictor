"""Train the XGBoost classifier for forward risk increase."""

from __future__ import annotations

import argparse
import pickle

import pandas as pd
from xgboost import XGBClassifier

from src.utils.config import ProjectConfig, load_config
from src.utils.exceptions import DataValidationError
from src.utils.io import read_parquet
from src.utils.paths import ensure_parent, get_paths


TARGET_COLUMN = "target_increase_12m"
DROP_COLUMNS = {
    "hex_id",
    "prediction_year",
    "risk_score_t_plus_1",
    "risk_delta_12m",
    TARGET_COLUMN,
}

MODEL_FEATURE_CANDIDATES = [
    "year",
    "risk_score_t",
    "heat_score",
    "vegetation_loss_score",
    "urban_expansion_score",
    "land_stress_score",
    "asset_exposure_score",
    "air_heat_change_1y",
    "air_heat_change_3y",
    "surface_heat_change_1y",
    "surface_heat_change_3y",
    "veg_loss_1y",
    "veg_loss_3y",
    "built_gain_1y",
    "built_gain_3y",
    "hot_days_count",
    "extreme_hot_days_count",
    "max_consecutive_hot_days",
    "temp_trend_3y",
    "air_temp_anomaly_vs_baseline",
    "air_temp_mean_annual",
    "air_temp_mean_summer",
    "building_density",
    "building_area_sum_m2",
]


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


def _positive_class_weight(target: pd.Series) -> float:
    """Compute scale_pos_weight = negative / positive."""
    positive_count = int((target == 1).sum())
    negative_count = int((target == 0).sum())
    if positive_count == 0:
        return 1.0
    return negative_count / positive_count


def _select_feature_columns(training: pd.DataFrame) -> list[str]:
    """Use a compact, curated feature set instead of every numeric column."""
    feature_columns = [
        column
        for column in MODEL_FEATURE_CANDIDATES
        if column in training.columns
        and column not in DROP_COLUMNS
        and pd.api.types.is_numeric_dtype(training[column])
    ]

    if not feature_columns:
        raise DataValidationError(
            "No valid model feature columns were found in the training table."
        )

    return feature_columns


def _build_model(
    *,
    n_estimators: int,
    scale_pos_weight: float,
    early_stopping_rounds: int | None = None,
) -> XGBClassifier:
    """Create the tuned XGBoost classifier."""
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=5,
        gamma=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds,
    )


def run(config: ProjectConfig | None = None) -> dict[str, object]:
    """Train the XGBoost classifier and persist the model artifact."""
    config = config or load_config()
    paths = get_paths()
    training = read_parquet(
        paths.training_table,
        "Training table is missing. Run `python -m src.cli build-training-data` first.",
    )
    training = _collapse_merge_duplicates(training)

    if training[TARGET_COLUMN].nunique(dropna=True) < 2:
        raise DataValidationError(
            "Training data must contain both positive and negative target_increase_12m classes."
        )

    if "year" not in training.columns:
        raise DataValidationError("Training table must contain a `year` column.")

    feature_columns = _select_feature_columns(training)
    X = training[feature_columns]
    y = training[TARGET_COLUMN].astype(int)

    years = sorted(training["year"].dropna().unique())
    if len(years) < 2:
        raise DataValidationError(
            "At least two distinct years are required for temporal validation."
        )

    validation_year = int(years[-1])
    train_mask = training["year"] < validation_year
    val_mask = training["year"] == validation_year

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_val = X.loc[val_mask]
    y_val = y.loc[val_mask]

    if X_train.empty or X_val.empty:
        raise DataValidationError(
            "Temporal split failed. Need earlier years for training and the latest year for validation."
        )

    if y_train.nunique(dropna=True) < 2:
        raise DataValidationError(
            "Training split must contain both positive and negative classes."
        )

    if y_val.nunique(dropna=True) < 2:
        raise DataValidationError(
            "Validation split must contain both positive and negative classes."
        )

    validation_scale_pos_weight = _positive_class_weight(y_train)
    staging_model = _build_model(
        n_estimators=2000,
        scale_pos_weight=validation_scale_pos_weight,
        early_stopping_rounds=50,
    )
    staging_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iteration = getattr(staging_model, "best_iteration", None)
    best_n_estimators = int(best_iteration) + 1 if best_iteration is not None else 2000

    full_scale_pos_weight = _positive_class_weight(y)
    model = _build_model(
        n_estimators=best_n_estimators,
        scale_pos_weight=full_scale_pos_weight,
        early_stopping_rounds=None,
    )
    model.fit(X, y, verbose=False)

    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "validation_year": validation_year,
        "best_n_estimators": best_n_estimators,
        "scale_pos_weight": full_scale_pos_weight,
    }

    ensure_parent(paths.model_path)
    with paths.model_path.open("wb") as handle:
        pickle.dump(artifact, handle)

    importance = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(paths.feature_importance, index=False)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the forward-risk XGBoost classifier.")
    parser.add_argument("--config", default=None, help="Path to the YAML config file.")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()