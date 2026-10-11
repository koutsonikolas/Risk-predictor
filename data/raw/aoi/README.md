# AOI Input

Place the Greater Phoenix boundary file at:

`data/raw/aoi/greater_phoenix.geojson`

Expected content:

- GeoJSON
- one or more polygon or multipolygon features
- CRS either declared or convertible to EPSG:4326

The pipeline will stop with a clear error if this file is missing or empty.
