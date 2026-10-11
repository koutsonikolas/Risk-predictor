"""Generate interpretable alert rows from scores and predictions."""

from __future__ import annotations

import argparse
import math

import pandas as pd

from src.utils.config import ProjectConfig, load_config
from src.utils.io import read_parquet, write_parquet
from src.utils.paths import get_paths


DRIVER_LABELS = {
    "heat_score": "Heat",
    "vegetation_loss_score": "Vegetation loss",
    "urban_expansion_score": "Urban expansion",
    "land_stress_score": "Land stress",
    "asset_exposure_score": "Asset exposure",
}


def _top_drivers(row: pd.Series) -> tuple[str, str]:
    ranked = sorted(
        DRIVER_LABELS,
        key=lambda column: float(row.get(column, 0.0)),
        reverse=True,
    )
    return DRIVER_LABELS[ranked[0]], DRIVER_LABELS[ranked[1]]


def _resolve_alert_count(config: ProjectConfig, eligible_count: int) -> int:
    """
    Resolve how many alerts to keep.

    Priority:
    1. alerts.alert_top_n
    2. alerts.alert_top_percent
    3. default to top 10%
    """
    alert_top_n = getattr(config.alerts, "alert_top_n", None)
    alert_top_percent = getattr(config.alerts, "alert_top_percent", 0.10)

    if eligible_count <= 0:
        return 0

    if alert_top_n is not None:
        return max(1, min(int(alert_top_n), eligible_count))

    return max(1, min(math.ceil(eligible_count * float(alert_top_percent)), eligible_count))


def run(config: ProjectConfig | None = None) -> pd.DataFrame:
    """Generate the current alert table and write parquet output."""
    config = config or load_config()
    paths = get_paths()

    scores = read_parquet(
        paths.risk_scores,
        "Risk scores are missing. Run `python -m src.cli score-risk` first.",
    )
    buildings = read_parquet(
        paths.buildings_features,
        "Building features are missing. Run `python -m src.cli ingest-buildings` first.",
    )
    predictions = read_parquet(
        paths.predictions_current,
        "Predictions are missing. Run `python -m src.cli predict-risk` first.",
    )

    latest_year = int(scores["year"].max())
    current = scores[scores["year"] == latest_year].copy()

    if current.empty:
        alerts = pd.DataFrame(
            columns=[
                "alert_rank",
                "hex_id",
                "current_risk_score",
                "predicted_risk_increase_prob",
                "primary_driver",
                "secondary_driver",
                "building_count",
                "alert_text",
            ]
        )
        write_parquet(alerts, paths.alerts_current)
        return alerts

    current = current.merge(
        predictions[["hex_id", "prob_risk_increase_12m"]],
        on="hex_id",
        how="left",
    )
    current = current.merge(
        buildings[["hex_id", "building_count"]],
        on="hex_id",
        how="left",
    )

    current["building_count"] = current["building_count"].fillna(0).astype(int)
    current["predicted_risk_increase_prob"] = current["prob_risk_increase_12m"].fillna(0.0)
    current["current_risk_score"] = current["risk_score"]

    current["primary_driver"], current["secondary_driver"] = zip(*current.apply(_top_drivers, axis=1))

    current["alert_text"] = current.apply(
        lambda row: (
            f"{row['primary_driver']} and {row['secondary_driver']} are the main drivers in this hex. "
            f"{row['building_count']} buildings are exposed, and the model estimates a "
            f"{row['predicted_risk_increase_prob']:.0%} probability of further risk increase over 12 months."
        ),
        axis=1,
    )

    eligible = current[current["building_count"] >= config.alerts.min_building_count].copy()

    if eligible.empty:
        alerts = pd.DataFrame(
            columns=[
                "alert_rank",
                "hex_id",
                "current_risk_score",
                "predicted_risk_increase_prob",
                "primary_driver",
                "secondary_driver",
                "building_count",
                "alert_text",
            ]
        )
        write_parquet(alerts, paths.alerts_current)
        return alerts

    eligible = eligible.sort_values(
        ["predicted_risk_increase_prob", "current_risk_score"],
        ascending=False,
    ).reset_index(drop=True)

    alert_count = _resolve_alert_count(config, len(eligible))
    alerts = eligible.head(alert_count).copy()
    alerts.insert(0, "alert_rank", range(1, len(alerts) + 1))

    alerts = alerts[
        [
            "alert_rank",
            "hex_id",
            "current_risk_score",
            "predicted_risk_increase_prob",
            "primary_driver",
            "secondary_driver",
            "building_count",
            "alert_text",
        ]
    ]

    write_parquet(alerts, paths.alerts_current)
    return alerts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the current alert table.")
    parser.add_argument("--config", default=None, help="Path to the YAML config file.")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()