"""Google Earth Engine initialization and conversion helpers."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from src.aoi.loader import load_aoi_geometry
from src.utils.config import ProjectConfig, load_config
from src.utils.exceptions import EarthEngineSetupError
from src.utils.logging_utils import get_logger


LOGGER = get_logger(__name__)


def initialize_earth_engine(config: ProjectConfig | None = None):
    """Initialize Earth Engine using local auth or a configured service account."""
    config = config or load_config()
    load_dotenv()

    try:
        import ee
    except ImportError as exc:
        raise EarthEngineSetupError(
            "earthengine-api is not installed. Run `pip install -e .` before using ingestion modules."
        ) from exc

    project = os.getenv("EE_PROJECT")
    opt_url = None
    if config.earth_engine.use_high_volume_endpoint:
        opt_url = "https://earthengine-highvolume.googleapis.com"

    try:
        ee.Initialize(project=project, opt_url=opt_url)
    except Exception as exc:
        raise EarthEngineSetupError(
            "Google Earth Engine is not initialized. Run `earthengine authenticate` and "
            "`python -c \"import ee; ee.Initialize()\"`, or provide a service account via "
            "GOOGLE_APPLICATION_CREDENTIALS and optionally EE_PROJECT."
        ) from exc

    LOGGER.info("Earth Engine initialized successfully.")
    return ee


def aoi_to_ee_geometry(config: ProjectConfig | None = None):
    """Convert the local AOI geometry into an Earth Engine geometry."""
    config = config or load_config()
    ee = initialize_earth_engine(config)
    geometry = load_aoi_geometry(config)
    return ee.Geometry(geometry.__geo_interface__)


def year_date_filter(ee: Any, year: int):
    """Return an Earth Engine date filter for a calendar year."""
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    return ee.Filter.date(start, end)


def seasonal_date_filter(ee: Any, year: int, months: list[int]):
    """Return an Earth Engine date filter covering the configured seasonal window."""
    ordered = sorted(months)
    start = ee.Date.fromYMD(year, ordered[0], 1)
    end_month = ordered[-1]
    end = ee.Date.fromYMD(year, end_month, 1).advance(1, "month")
    return ee.Filter.date(start, end)
