"""Configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.utils.exceptions import ConfigurationError
from src.utils.paths import PROJECT_ROOT


@dataclass(frozen=True)
class YearsConfig:
    start: int
    end: int

    @property
    def values(self) -> list[int]:
        return list(range(self.start, self.end + 1))


@dataclass(frozen=True)
class InputConfig:
    aoi_path: str = "data/raw/aoi/greater_phoenix.geojson"
    buildings_path: str = "data/raw/buildings/buildings.geojson"


@dataclass(frozen=True)
class EarthEngineConfig:
    use_high_volume_endpoint: bool = False
    landsat_scale: int = 30
    nlcd_scale: int = 30
    dynamic_world_scale: int = 10
    era5_scale: int = 11132


@dataclass(frozen=True)
class AlertConfig:
    risk_score_threshold: float = 65.0
    probability_threshold: float = 0.60
    min_building_count: int = 1


@dataclass(frozen=True)
class ProjectConfig:
    project_name: str
    aoi_name: str
    analysis_crs: str
    hex_resolution: int
    years: YearsConfig
    season_months: list[int]
    prediction_horizon_months: int
    score_refresh_months: int
    baseline_years: list[int]
    inputs: InputConfig = field(default_factory=InputConfig)
    earth_engine: EarthEngineConfig = field(default_factory=EarthEngineConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)

    @property
    def aoi_path(self) -> Path:
        return PROJECT_ROOT / self.inputs.aoi_path

    @property
    def buildings_path(self) -> Path:
        return PROJECT_ROOT / self.inputs.buildings_path

    @property
    def analysis_years(self) -> list[int]:
        return self.years.values


def _merge_dataclass_defaults(default_cls: type[Any], values: dict[str, Any] | None) -> Any:
    values = values or {}
    return default_cls(**values)


def load_config(config_path: str | Path | None = None) -> ProjectConfig:
    """Load the main YAML config into a typed object."""
    load_dotenv()
    selected = config_path or os.getenv("EARTHRISK_CONFIG") or "configs/phoenix.yaml"
    path = Path(selected)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise ConfigurationError(
            f"Config file not found at {path}. Add it or set EARTHRISK_CONFIG to a valid path."
        )

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    try:
        years = YearsConfig(**raw["years"])
        inputs = _merge_dataclass_defaults(InputConfig, raw.get("inputs"))
        earth_engine = _merge_dataclass_defaults(EarthEngineConfig, raw.get("earth_engine"))
        alerts = _merge_dataclass_defaults(AlertConfig, raw.get("alerts"))
        config = ProjectConfig(
            project_name=raw["project_name"],
            aoi_name=raw["aoi_name"],
            analysis_crs=raw["analysis_crs"],
            hex_resolution=int(raw["hex_resolution"]),
            years=years,
            season_months=list(raw["season_months"]),
            prediction_horizon_months=int(raw["prediction_horizon_months"]),
            score_refresh_months=int(raw["score_refresh_months"]),
            baseline_years=list(raw["baseline_years"]),
            inputs=inputs,
            earth_engine=earth_engine,
            alerts=alerts,
        )
    except KeyError as exc:
        raise ConfigurationError(f"Missing required config key: {exc}") from exc
    except TypeError as exc:
        raise ConfigurationError(f"Invalid config structure in {path}: {exc}") from exc

    if config.years.start > config.years.end:
        raise ConfigurationError("years.start must be less than or equal to years.end.")
    if not config.season_months:
        raise ConfigurationError("season_months must contain at least one month.")
    if not config.baseline_years:
        raise ConfigurationError("baseline_years must contain at least one year.")
    return config
