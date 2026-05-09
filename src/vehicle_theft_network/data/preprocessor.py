from typing import Sequence

import pandas as pd
from unidecode import unidecode


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.drop_duplicates(subset=["ANO_BO", "NUM_BO", "NOME_DELEGACIA"])
        .reset_index(drop=True)
    )


def select_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return df[list(columns)]


def _normalize_text(text: object) -> str:
    return unidecode(text).upper().strip() if isinstance(text, str) else "NAO INFORMADO"


def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["BAIRRO"] = df["BAIRRO"].apply(_normalize_text)
    df["CIDADE"] = df["CIDADE"].apply(_normalize_text)
    return df


def remove_invalid_coords(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["LATITUDE"] != 0)
        & df["LATITUDE"].notna()
        & df["LONGITUDE"].notna()
    ]


def filter_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    df = df.copy()
    df["DATA_OCORRENCIA_BO"] = pd.to_datetime(df["DATA_OCORRENCIA_BO"], errors="coerce")
    df = df.dropna(subset=["DATA_OCORRENCIA_BO"])
    mask = (df["DATA_OCORRENCIA_BO"] >= start) & (df["DATA_OCORRENCIA_BO"] <= end)
    return df[mask].reset_index(drop=True)


def combine_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["HORA_OCORRENCIA"] = pd.to_datetime(
        df["HORA_OCORRENCIA"], format="%H:%M:%S", errors="coerce"
    ).dt.time
    df["DATA_HORA"] = pd.to_datetime(
        df["DATA_OCORRENCIA_BO"].astype(str) + " " + df["HORA_OCORRENCIA"].astype(str),
        errors="coerce",
    )
    return df


def preprocess(
    df: pd.DataFrame,
    columns: Sequence[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    df = drop_duplicates(df)
    df = select_columns(df, columns)
    df = normalize_strings(df)
    df = remove_invalid_coords(df)
    df = filter_dates(df, start_date, end_date)
    df = combine_datetime(df)
    return df
