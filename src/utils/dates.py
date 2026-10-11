"""Date helpers for yearly and seasonal windows."""

from __future__ import annotations

import calendar
from datetime import date


def year_window(year: int) -> tuple[str, str]:
    """Return ISO start and end dates for a calendar year."""
    return date(year, 1, 1).isoformat(), date(year, 12, 31).isoformat()


def season_window(year: int, months: list[int]) -> tuple[str, str]:
    """Return ISO start and end dates spanning the configured season."""
    ordered = sorted(months)
    start = date(year, ordered[0], 1)
    last_month = ordered[-1]
    end_day = calendar.monthrange(year, last_month)[1]
    end = date(year, last_month, end_day)
    return start.isoformat(), end.isoformat()
