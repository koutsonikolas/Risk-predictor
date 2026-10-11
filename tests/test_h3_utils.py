from shapely.geometry import Polygon

from src.utils.h3_utils import polygon_to_cells


def test_polygon_to_cells_returns_values():
    polygon = Polygon(
        [
            (-112.2, 33.3),
            (-112.1, 33.3),
            (-112.1, 33.4),
            (-112.2, 33.4),
            (-112.2, 33.3),
        ]
    )
    assert polygon_to_cells(polygon, 8)
