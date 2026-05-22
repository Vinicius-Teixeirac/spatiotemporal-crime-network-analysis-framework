from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def build_time_series(
    cell_data_dict: Dict[Tuple[float, float], Dict[str, Any]],
    freq: str = "h",
) -> pd.DataFrame:
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
    non_zero = df.astype(bool).sum()
    return df[non_zero[non_zero >= min_events].index]


def to_binary_event_series(df: pd.DataFrame) -> np.ndarray:
    return np.where(np.array(df) > 0, 1, 0)
