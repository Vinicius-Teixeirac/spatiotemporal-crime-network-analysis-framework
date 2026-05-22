"""Complex network analysis of vehicle theft in São Paulo, ICMC/USP, 2023."""

from vehicle_theft_network.config import (
    DataConfig,
    EventSyncConfig,
    GridConfig,
    TimeSeriesConfig,
)
from vehicle_theft_network.data.loader import (
    load_npy,
    load_pickle,
    load_processed,
    load_raw_data,
    save_npy,
    save_pickle,
)
from vehicle_theft_network.data.preprocessor import preprocess
from vehicle_theft_network.event_sync.analyzer import (
    build_adjacency_matrix,
    build_event_series,
    compute_significance,
)
from vehicle_theft_network.grid.builder import aggregate_cells, build_grid_cells
from vehicle_theft_network.network.builder import build_network, get_coord_labels
from vehicle_theft_network.network.metrics import (
    compute_local_metrics,
    compute_structural_metrics,
    detect_communities,
)
from vehicle_theft_network.timeseries.builder import (
    build_time_series,
    filter_time_series,
    to_binary_event_series,
)

__all__ = [
    # Config
    "DataConfig",
    "EventSyncConfig",
    "GridConfig",
    "TimeSeriesConfig",
    # I/O
    "load_raw_data",
    "load_processed",
    "load_pickle",
    "save_pickle",
    "load_npy",
    "save_npy",
    # Preprocessing
    "preprocess",
    # Grid
    "build_grid_cells",
    "aggregate_cells",
    # Time series
    "build_time_series",
    "filter_time_series",
    "to_binary_event_series",
    # Event synchronization
    "build_event_series",
    "compute_significance",
    "build_adjacency_matrix",
    # Network
    "get_coord_labels",
    "build_network",
    "compute_structural_metrics",
    "compute_local_metrics",
    "detect_communities",
]
