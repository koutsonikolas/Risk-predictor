"""Compatibility helpers for H3 operations."""

from __future__ import annotations

from collections.abc import Iterable

import h3
from pyproj import Geod
from shapely.geometry import MultiPolygon, Polygon, mapping, shape


GEOD = Geod(ellps="WGS84")


def _iter_polygons(geometry: Polygon | MultiPolygon) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    return list(geometry.geoms)


def polygon_to_cells(geometry: Polygon | MultiPolygon, resolution: int) -> list[str]:
    """Fill a polygon with H3 cells while supporting h3-py 3.x and 4.x."""
    cell_ids: set[str] = set()
    for polygon in _iter_polygons(geometry):
        geojson = mapping(polygon)
        if hasattr(h3, "geo_to_cells"):
            cell_ids.update(h3.geo_to_cells(geojson, resolution))
        elif hasattr(h3, "polyfill"):
            cell_ids.update(h3.polyfill(geojson, resolution, geo_json_conformant=True))
        else:
            shell = [(lat, lon) for lon, lat in geojson["coordinates"][0]]
            holes = [[(lat, lon) for lon, lat in ring] for ring in geojson["coordinates"][1:]]
            latlng_poly = h3.LatLngPoly(shell, holes)
            cell_ids.update(h3.polygon_to_cells(latlng_poly, resolution))
    return sorted(cell_ids)


def cell_to_polygon(cell_id: str) -> Polygon:
    """Convert an H3 cell to a shapely polygon."""
    if hasattr(h3, "cell_to_boundary"):
        boundary = h3.cell_to_boundary(cell_id)
        coordinates = [(lon, lat) for lat, lon in boundary]
    else:
        coordinates = h3.h3_to_geo_boundary(cell_id, geo_json=True)
    return Polygon(coordinates)


def cell_to_centroid(cell_id: str) -> tuple[float, float]:
    """Return longitude and latitude of a cell centroid."""
    if hasattr(h3, "cell_to_latlng"):
        lat, lon = h3.cell_to_latlng(cell_id)
    else:
        lat, lon = h3.h3_to_geo(cell_id)
    return lon, lat


def polygon_area_m2(geometry: Polygon | MultiPolygon) -> float:
    """Compute geodesic area in square meters."""
    area, _ = GEOD.geometry_area_perimeter(geometry)
    return abs(area)


def cell_area_m2(cell_id: str) -> float:
    """Compute the geodesic area of an H3 cell."""
    return polygon_area_m2(cell_to_polygon(cell_id))


def feature_from_cell(cell_id: str) -> dict[str, object]:
    """Return a GeoJSON-like feature for a cell."""
    polygon = cell_to_polygon(cell_id)
    return {
        "type": "Feature",
        "properties": {"hex_id": cell_id},
        "geometry": mapping(polygon),
    }


def geometry_from_geo_interface(obj: dict[str, object]) -> Polygon | MultiPolygon:
    """Convert a GeoJSON-like mapping to a shapely geometry."""
    return shape(obj)
