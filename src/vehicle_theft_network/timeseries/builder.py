from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def build_time_series(
    cell_data_dict: Dict[Tuple[float, float], Dict[str, Any]],
    freq: str = "h",
) -> pd.DataFrame:
    """Build a regular event-count time series for every spatial cell.

    Parameters
    ----------
    cell_data_dict:
        Mapping from ``(lon, lat)`` centroid to cell metadata, as returned by
        :func:`~vehicle_theft_network.grid.builder.aggregate_cells`.
    freq:
        Pandas offset alias for the time grid (e.g. ``"h"`` for hourly).

    Returns
    -------
    pd.DataFrame
        Index is the regular date range; columns are ``(lon, lat)`` coordinate
        tuples; values are event counts per period.
    """
    if not cell_data_dict:
        raise ValueError("cell_data_dict is empty: no cells to build a time series from.")

    all_dates = [
        dt
        for info in cell_data_dict.values()
        for dt in info["DATA"]
        if pd.notna(dt)
    ]
    date_range = pd.date_range(min(all_dates), max(all_dates), freq=freq)

    data: Dict[Any, Any] = {"date": date_range}
    for coords, info in cell_data_dict.items():
        raw = [dt for dt in info["DATA"] if pd.notna(dt)]
        valid = pd.to_datetime(pd.Series(raw, dtype="object"))
        counts = valid.dt.floor(freq).value_counts().reindex(date_range, fill_value=0)
        data[coords] = counts.values

    df = pd.DataFrame(data).set_index("date")
    df.index.name = None
    return df


def filter_time_series(df: pd.DataFrame, min_events: int = 10) -> pd.DataFrame:
    """Keep only columns (cells) that have at least ``min_events`` non-zero periods."""
    non_zero = df.astype(bool).sum()
    return df[non_zero[non_zero >= min_events].index]


def to_binary_event_series(df: pd.DataFrame) -> np.ndarray:
    """Convert event-count matrix to a binary (0/1) event matrix.

    Any count > 0 becomes 1; zero counts remain 0.
    """
    return np.where(np.array(df) > 0, 1, 0)
