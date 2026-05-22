from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent


@dataclass
class DataConfig:
    """Paths and column selection for raw and processed data.

    ``sheets`` is only used when ``raw_data_path`` points to an Excel file.
    For Parquet sources the field is ignored.
    """

    raw_data_path: Path = ROOT_DIR / "data" / "VeiculosSubtraidos.parquet"
    processed_parquet: Path = ROOT_DIR / "data" / "roubo_veiculo_selecionado.parquet"
    grid_cells_path: Path = ROOT_DIR / "eventsync17000" / "grid_cells.pkl"
    cell_data_dict_path: Path = ROOT_DIR / "eventsync17000" / "cell_data_dict.pkl"
    time_series_path: Path = ROOT_DIR / "hourly_filtered_time_series_df"
    monte_carlo_path: Path = ROOT_DIR / "monteCarlosimulations.npy"
    sheets: tuple[str, ...] = ("VEICULOS_2022", "VEICULOS_2023", "VEICULOS_2024")
    columns_to_keep: tuple[str, ...] = (
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
    """Spatial grid resolution in decimal degrees (~200 m × 200 m at São Paulo's latitude)."""

    grid_size: float = 0.002

    def __post_init__(self) -> None:
        if self.grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {self.grid_size}")


@dataclass
class TimeSeriesConfig:
    """Parameters controlling the hourly event time series construction."""

    frequency: str = "h"
    min_events: int = 10

    def __post_init__(self) -> None:
        if self.min_events < 1:
            raise ValueError(f"min_events must be >= 1, got {self.min_events}")


@dataclass
class EventSyncConfig:
    """Parameters for the Event Synchronization and Monte Carlo significance test.

    ``symmetrization`` controls the network directionality:
    ``"symmetric"`` -> undirected, ``"directed"`` -> directed.
    """

    taumax: int = 5
    n_surr: int = 1000
    significance_level: float = 0.95
    symmetrization: str = "symmetric"

    def __post_init__(self) -> None:
        if self.taumax < 1:
            raise ValueError(f"taumax must be >= 1, got {self.taumax}")
        if self.n_surr < 1:
            raise ValueError(f"n_surr must be >= 1, got {self.n_surr}")
        if not (0.0 < self.significance_level < 1.0):
            raise ValueError(
                f"significance_level must be in (0, 1), got {self.significance_level}"
            )
        if self.symmetrization not in {"symmetric", "directed"}:
            raise ValueError(
                f"symmetrization must be 'symmetric' or 'directed', got {self.symmetrization!r}"
            )
