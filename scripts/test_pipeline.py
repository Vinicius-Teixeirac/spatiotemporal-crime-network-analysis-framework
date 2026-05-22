"""Smoke-test for the vehicle-theft-network pipeline.

Generates 500 synthetic records that mimic the SSP-SP schema, overrides
configuration for speed, and runs every pipeline stage end-to-end.

pyunicorn is *not* required: when it is absent the Monte Carlo stage is
skipped and a random adjacency matrix is used in its place.

Usage:
    uv run python scripts/test_pipeline.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vehicle_theft_network.data.preprocessor import preprocess
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

# ---------------------------------------------------------------------------
# Test configuration: smaller values so the full pipeline finishes fast
# ---------------------------------------------------------------------------
GRID_SIZE = 0.05    # ~5 km cells -> ~20 cells from 500 Sao-Paulo-area points
MIN_EVENTS = 2      # low bar so cells survive the time-series filter
N_SURR = 5          # tiny Monte Carlo (or skipped when pyunicorn absent)
SIG_LEVEL = 0.5     # 50 % threshold -> denser test graph
N_RECORDS = 500

LAT_MIN, LAT_MAX = -23.65, -23.45
LON_MIN, LON_MAX = -46.75, -46.50
DATE_MIN, DATE_MAX = "2022-01-01", "2022-01-31"

COLUMNS_TO_KEEP = (
    "BAIRRO", "CIDADE", "LATITUDE", "LONGITUDE",
    "DATA_OCORRENCIA_BO", "HORA_OCORRENCIA", "DESCR_MARCA_VEICULO",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tick(label: str) -> float:
    print(f"\n>>> {label}")
    return time.perf_counter()


def _tock(t0: float) -> None:
    print(f"    done in {time.perf_counter() - t0:.2f}s")


def make_synthetic_data(n: int = N_RECORDS, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame that matches the SSP-SP column schema."""
    rng = np.random.default_rng(seed)

    date_range = pd.date_range(DATE_MIN, DATE_MAX, freq="D")
    date_idx = rng.integers(0, len(date_range), n)
    dates = date_range[date_idx].strftime("%Y-%m-%d")

    hours = rng.integers(0, 24, n)
    hora = [f"{h:02d}:00:00" for h in hours]

    brands = ["FIAT", "VOLKSWAGEN", "CHEVROLET", "HONDA", "TOYOTA"]
    cities = ["SAO PAULO", "GUARULHOS", "OSASCO"]
    bairros = ["CENTRO", "VILA MADALENA", "BELA VISTA", "LAPA", "MOOCA"]
    delegacias = ["1 DP", "2 DP", "3 DP", "4 DP"]

    return pd.DataFrame({
        # Unique per record so deduplication keeps all 500
        "ANO_BO":              [f"2022{i:06d}" for i in range(n)],
        "NUM_BO":              [f"{i:06d}" for i in range(n)],
        "NOME_DELEGACIA":      rng.choice(delegacias, n),
        "BAIRRO":              rng.choice(bairros, n),
        "CIDADE":              rng.choice(cities, n),
        "LATITUDE":            rng.uniform(LAT_MIN, LAT_MAX, n),
        "LONGITUDE":           rng.uniform(LON_MIN, LON_MAX, n),
        "DATA_OCORRENCIA_BO":  dates,
        "HORA_OCORRENCIA":     hora,
        "DESCR_MARCA_VEICULO": rng.choice(brands, n),
    })


def _random_adjacency(n_cells: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    upper = np.triu(rng.integers(0, 2, (n_cells, n_cells)), k=1)
    return (upper + upper.T).astype(np.int8)


def build_adjacency(event_array: np.ndarray, n_cells: int) -> np.ndarray:
    try:
        from vehicle_theft_network.event_sync.analyzer import (
            build_adjacency_matrix,
            build_event_series,
            compute_significance,
        )
        ev = build_event_series(event_array, taumax=2)
        counts = compute_significance(ev, n_surr=N_SURR, symmetrization="symmetric")
        return build_adjacency_matrix(counts, n_surr=N_SURR, significance_level=SIG_LEVEL)
    except ImportError:
        print("    [pyunicorn not installed; using random adjacency matrix as fallback]")
        return _random_adjacency(n_cells)


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run_smoke_test() -> None:
    t_overall = time.perf_counter()
    failed: list[str] = []

    # ------------------------------------------------------------------
    # Stage 1 - Preprocessing
    # ------------------------------------------------------------------
    t = _tick("Stage 1 - Preprocessing")
    raw = make_synthetic_data()
    dados = preprocess(raw, columns=COLUMNS_TO_KEEP, start_date=DATE_MIN, end_date=DATE_MAX)
    if len(dados) == 0:
        failed.append("preprocess: no records survived")
    else:
        print(f"    {len(raw)} raw -> {len(dados)} after preprocessing")
    _tock(t)

    if not dados.empty:
        # ------------------------------------------------------------------
        # Stage 2 - Spatial grid
        # ------------------------------------------------------------------
        t = _tick("Stage 2 - Spatial grid")
        grid_cells = build_grid_cells(dados, grid_size=GRID_SIZE)
        cell_data_dict = aggregate_cells(grid_cells)
        if len(cell_data_dict) == 0:
            failed.append("grid: no cells produced")
        else:
            print(f"    {len(cell_data_dict)} unique cells")
        _tock(t)

        # ------------------------------------------------------------------
        # Stage 3 - Time series
        # ------------------------------------------------------------------
        t = _tick("Stage 3 - Time series")
        ts_df = build_time_series(cell_data_dict, freq="h")
        ts_filtered = filter_time_series(ts_df, min_events=MIN_EVENTS)
        event_array = to_binary_event_series(ts_filtered)
        n_cells = ts_filtered.shape[1]
        if n_cells == 0:
            failed.append("timeseries: all cells dropped - lower min_events")
        else:
            print(f"    {ts_df.shape[1]} cells before filter -> {n_cells} retained")
            print(f"    event_array shape: {event_array.shape}")
        _tock(t)

        if n_cells > 0:
            # ------------------------------------------------------------------
            # Stage 4 - Event Synchronization / adjacency matrix
            # ------------------------------------------------------------------
            t = _tick("Stage 4 - Event Synchronization")
            adjacency = build_adjacency(event_array, n_cells)
            n_edges = int(adjacency.sum()) // 2
            print(f"    adjacency shape: {adjacency.shape}  |  {n_edges} edges")
            _tock(t)

            # ------------------------------------------------------------------
            # Stage 5 - Network construction
            # ------------------------------------------------------------------
            t = _tick("Stage 5 - Network construction")
            coord_labels = get_coord_labels(ts_filtered)
            G = build_network(adjacency, coord_labels)
            print(f"    nodes: {G.number_of_nodes()}  edges: {G.number_of_edges()}")
            _tock(t)

            # ------------------------------------------------------------------
            # Stage 6 - Structural metrics
            # ------------------------------------------------------------------
            t = _tick("Stage 6 - Structural metrics")
            structural = compute_structural_metrics(G)
            for k, v in structural.items():
                print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")
            _tock(t)

            # ------------------------------------------------------------------
            # Stage 7 - Local metrics
            # ------------------------------------------------------------------
            t = _tick("Stage 7 - Local metrics")
            local = compute_local_metrics(G)
            hub = max(local["degree_centrality"], key=local["degree_centrality"].get)
            print(f"    hub node: {coord_labels[hub]}  degree={local['degree'][hub]}")
            _tock(t)

            # ------------------------------------------------------------------
            # Stage 8 - Community detection
            # ------------------------------------------------------------------
            t = _tick("Stage 8 - Community detection (Girvan-Newman)")
            if G.number_of_edges() < 1 or G.number_of_nodes() < 2:
                print("    skipped - graph has no edges to cut")
            else:
                communities = detect_communities(G)
                print(f"    {len(communities)} communities detected")
            _tock(t)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t_overall
    print(f"\n{'='*55}")
    if failed:
        print(f"SMOKE TEST FAILED in {elapsed:.1f}s")
        for msg in failed:
            print(f"  FAIL: {msg}")
        sys.exit(1)
    else:
        print(f"SMOKE TEST PASSED in {elapsed:.1f}s")


if __name__ == "__main__":
    run_smoke_test()
