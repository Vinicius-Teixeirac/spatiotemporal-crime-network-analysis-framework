"""Pipeline orchestrator.

Stages
------
1. preprocess  - load raw Excel sheets, clean and filter records, export Parquet
2. grid        - build spatial grid, aggregate events per cell, export pickles
3. timeseries  - build hourly time series, apply event-count filter, export
4. network     - load Monte Carlo results, build adjacency matrix, analyse graph
5. visualize   - produce interactive Folium maps and save as HTML

Run a single stage:
    python scripts/run_pipeline.py --stage preprocess

Run the full pipeline (stages 1-4; stage 5 is opt-in):
    python scripts/run_pipeline.py
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vehicle_theft_network.config import (
    DataConfig,
    EventSyncConfig,
    GridConfig,
    TimeSeriesConfig,
)
from vehicle_theft_network.data.loader import (
    load_npy,
    load_pickle,
    load_raw_data,
    save_npy,
    save_pickle,
)
from vehicle_theft_network.data.preprocessor import preprocess
from vehicle_theft_network.event_sync.analyzer import build_adjacency_matrix
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
from vehicle_theft_network.visualization.map_utils import (
    build_centroid_kdtree,
    generate_centroid_polygon_dict,
    plot_communities_on_map,
    plot_metric_map,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

data_cfg = DataConfig()
grid_cfg = GridConfig()
ts_cfg = TimeSeriesConfig()
ev_cfg = EventSyncConfig()

OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# Stage 1: Preprocessing
# --------------------------------------------------------------------------- #

def stage_preprocess() -> pd.DataFrame:
    log.info("Stage 1/4: Loading and preprocessing raw data...")
    raw = load_raw_data(data_cfg.raw_data_path, data_cfg.sheets)
    dados = preprocess(raw, data_cfg.columns_to_keep, data_cfg.start_date, data_cfg.end_date)
    dados.to_parquet(data_cfg.processed_parquet, index=False)
    log.info("%d records saved to %s", len(dados), data_cfg.processed_parquet)
    return dados


# --------------------------------------------------------------------------- #
# Stage 2: Grid
# --------------------------------------------------------------------------- #

def stage_grid(dados: pd.DataFrame):
    log.info("Stage 2/4: Building spatial grid and aggregating cells...")
    grid_cells = build_grid_cells(dados, grid_cfg.grid_size)
    cell_data_dict = aggregate_cells(grid_cells)
    save_pickle(grid_cells, data_cfg.grid_cells_path)
    save_pickle(cell_data_dict, data_cfg.cell_data_dict_path)
    log.info("%d unique cells; pickles saved.", len(cell_data_dict))
    return grid_cells, cell_data_dict


# --------------------------------------------------------------------------- #
# Stage 3: Time series
# --------------------------------------------------------------------------- #

def stage_timeseries(cell_data_dict):
    log.info("Stage 3/4: Building and filtering time series...")
    ts_df = build_time_series(cell_data_dict, freq=ts_cfg.frequency)
    ts_filtered = filter_time_series(ts_df, min_events=ts_cfg.min_events)
    event_series = to_binary_event_series(ts_filtered)

    ts_filtered.to_pickle(data_cfg.time_series_path)
    save_npy(event_series, OUTPUTS / "event_series.npy")
    log.info("%d cells retained; time series saved.", ts_filtered.shape[1])
    return ts_filtered, event_series


# --------------------------------------------------------------------------- #
# Stage 4: Network analysis
# --------------------------------------------------------------------------- #

def stage_network(ts_filtered: pd.DataFrame):
    log.info("Stage 4/4: Building complex network and computing metrics...")
    monte_carlo = load_npy(data_cfg.monte_carlo_path)
    adjacency = build_adjacency_matrix(
        monte_carlo, ev_cfg.n_surr, ev_cfg.significance_level
    )
    coord_labels = get_coord_labels(ts_filtered)
    G = build_network(adjacency, coord_labels)

    structural = compute_structural_metrics(G)
    local = compute_local_metrics(G)
    communities = detect_communities(G)

    log.info("Structural metrics:")
    for k, v in structural.items():
        log.info("  %s: %s", k, f"{v:.6f}" if isinstance(v, float) else v)

    hub = max(local["degree_centrality"], key=local["degree_centrality"].get)
    log.info("Hub node: %s", coord_labels[hub])
    return G, coord_labels, local, communities


# --------------------------------------------------------------------------- #
# Stage 5: Visualisation (opt-in)
# --------------------------------------------------------------------------- #

def stage_visualize(G, coord_labels, local, communities):
    log.info("Stage 5/5: Generating interactive maps...")
    grid_cells = load_pickle(data_cfg.grid_cells_path)
    cell_data_dict = load_pickle(data_cfg.cell_data_dict_path)

    polygon_dict = generate_centroid_polygon_dict(grid_cells)
    kdtree, centroids = build_centroid_kdtree(polygon_dict)

    communities_map = plot_communities_on_map(
        communities, coord_labels, polygon_dict, cell_data_dict, kdtree, centroids
    )
    communities_map.save(str(OUTPUTS / "communities.html"))

    degree_map = plot_metric_map(
        coord_labels=coord_labels,
        metric_values=local["degree"],
        polygon_dict=polygon_dict,
        cell_data_dict=cell_data_dict,
        kdtree=kdtree,
        centroids=centroids,
        visualization_type="polygon",
        metric_name="Degree",
        metric_precision=0,
    )
    degree_map.save(str(OUTPUTS / "degree_map.html"))
    log.info("Maps saved to %s/", OUTPUTS)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

STAGES = {
    "preprocess": stage_preprocess,
    "grid": stage_grid,
    "timeseries": stage_timeseries,
    "network": stage_network,
    "visualize": stage_visualize,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Vehicle theft complex network pipeline")
    parser.add_argument(
        "--stage",
        choices=list(STAGES.keys()),
        default=None,
        help="Run a single stage. Omit to run the full pipeline (1-4).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Also run the visualisation stage (slow).",
    )
    args = parser.parse_args()

    if args.stage:
        log.info("Running stage: %s", args.stage)
        STAGES[args.stage]()
        return

    dados = stage_preprocess()
    grid_cells, cell_data_dict = stage_grid(dados)
    ts_filtered, _ = stage_timeseries(cell_data_dict)
    G, coord_labels, local, communities = stage_network(ts_filtered)

    if args.visualize:
        stage_visualize(G, coord_labels, local, communities)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
