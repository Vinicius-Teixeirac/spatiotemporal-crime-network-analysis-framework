import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def load_raw_data(path: Path, sheets: Sequence[str] = ()) -> pd.DataFrame:
    """Load raw event data from Excel (.xlsx) or Parquet (.parquet / .pq).

    For Excel files, ``sheets`` lists the sheet names to concatenate.
    For Parquet files, the argument is ignored and the file is read directly.
    """
    suffix = Path(path).suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    frames = [pd.read_excel(path, sheet_name=sheet) for sheet in sheets]
    return pd.concat(frames, ignore_index=True)


def load_processed(path: Path) -> pd.DataFrame:
    """Load the preprocessed dataset from Parquet or CSV."""
    suffix = Path(path).suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    df = pd.read_csv(path)
    return df.drop(columns=["Unnamed: 0"], errors="ignore")


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def save_npy(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
