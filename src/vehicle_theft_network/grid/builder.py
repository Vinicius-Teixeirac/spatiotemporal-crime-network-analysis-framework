from typing import Any, Dict, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

# Decimal places used to round cell centroids when building the coordinate key.
# 7 decimals (~1 cm precision), fine enough to be collision-free at 0.002 deg grid size.
_CENTROID_DECIMALS = 7


def _cell_indices(lat: float, lon: float, grid_size: float) -> Tuple[int, int]:
    """Return the (row, col) grid index for a given coordinate pair."""
    return (int(lat // grid_size), int(lon // grid_size))


def _cell_polygon(row: int, col: int, grid_size: float) -> Polygon:
    """Build the bounding-box Polygon for a grid cell identified by (row, col)."""
    return Polygon([
        (col * grid_size, row * grid_size),
        ((col + 1) * grid_size, row * grid_size),
        ((col + 1) * grid_size, (row + 1) * grid_size),
        (col * grid_size, (row + 1) * grid_size),
    ])


def build_grid_cells(dados: pd.DataFrame, grid_size: float) -> gpd.GeoDataFrame:
    """Assign every theft record to a spatial grid cell.

    Returns a GeoDataFrame with one row per record, each annotated with the
    bounding-box Polygon of the cell it belongs to.
    """
    rows = []
    for _, record in dados.iterrows():
        r, c = _cell_indices(record["LATITUDE"], record["LONGITUDE"], grid_size)
        rows.append({
            "geometry": _cell_polygon(r, c, grid_size),
            "BAIRRO": record["BAIRRO"],
            "CIDADE": record["CIDADE"],
            "LATITUDE": record["LATITUDE"],
            "LONGITUDE": record["LONGITUDE"],
            "DATA": record["DATA_HORA"],
            "MODELO": record["DESCR_MARCA_VEICULO"],
        })
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def aggregate_cells(
    grid_cells: gpd.GeoDataFrame,
) -> Dict[Tuple[float, float], Dict[str, Any]]:
    """Group all theft events by spatial cell.

    Keys are ``(longitude, latitude)`` of the cell centroid, rounded to
    ``_CENTROID_DECIMALS`` decimal places.  Values are dicts with:
    - ``"DATA"``: list of event timestamps
    - ``"CIDADE"``: set of city names that fall in this cell
    - ``"BAIRRO"``: set of neighbourhood names that fall in this cell
    """
    cell_data: Dict[Tuple[float, float], Dict[str, Any]] = {}
    for _, row in grid_cells.iterrows():
        centroid = row["geometry"].centroid
        key = (round(centroid.x, _CENTROID_DECIMALS), round(centroid.y, _CENTROID_DECIMALS))
        if key in cell_data:
            cell_data[key]["DATA"].append(row["DATA"])
            cell_data[key]["CIDADE"].add(row["CIDADE"])
            cell_data[key]["BAIRRO"].add(row["BAIRRO"])
        else:
            cell_data[key] = {
                "DATA": [row["DATA"]],
                "CIDADE": {row["CIDADE"]},
                "BAIRRO": {row["BAIRRO"]},
            }
    return cell_data
