from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent


@dataclass
class DataConfig:
    # Parquet is the default; pass an .xlsx path + sheets for the original Excel source
    raw_data_path: Path = ROOT_DIR / "data" / "VeiculosSubtraidos.parquet"
    processed_parquet: Path = ROOT_DIR / "data" / "roubo_veiculo_selecionado.parquet"
    grid_cells_path: Path = ROOT_DIR / "eventsync17000" / "grid_cells.pkl"
    cell_data_dict_path: Path = ROOT_DIR / "eventsync17000" / "cell_data_dict.pkl"
    time_series_path: Path = ROOT_DIR / "hourly_filtered_time_series_df"
    monte_carlo_path: Path = ROOT_DIR / "monteCarlosimulations.npy"
    # sheets is only used when raw_data_path is an Excel file
    sheets: tuple = ("VEICULOS_2022", "VEICULOS_2023", "VEICULOS_2024")
    columns_to_keep: tuple = (
        "BAIRRO",
        "CIDADE",
        "LATITUDE",
        "LONGITUDE",
        "DATA_OCORRENCIA_BO",
        "HORA_OCORRENCIA",
        "DESCR_MARCA_VEICULO",
    )
    start_date: str = "2022-01-01"
    end_date: str = "2023-12-31"


@dataclass
class GridConfig:
    # ~200 m × 200 m cells at São Paulo's latitude
    grid_size: float = 0.002


@dataclass
class TimeSeriesConfig:
    frequency: str = "h"
    min_events: int = 10


@dataclass
class EventSyncConfig:
    taumax: int = 5
    n_surr: int = 1000
    significance_level: float = 0.95
    # "symmetric" builds an undirected network; "directed" builds a directed one
    symmetrization: str = "symmetric"
