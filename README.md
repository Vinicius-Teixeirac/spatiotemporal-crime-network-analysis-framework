# Vehicle Theft Network
### A Spatiotemporal Complex-Network Framework for Crime Analysis

> Developed as part of an undergraduate thesis (Trabalho de Conclusão de Curso) — Bachelor in Applied Mathematics and Scientific Computing  
> Instituto de Ciências Matemáticas e de Computação — ICMC/USP, São Carlos, 2023  
> **Author:** Vinícius Teixeira de Carvalho Freitas  
> **Advisor:** Prof. Dr. Luis Gustavo Nonato  
> **Co-advisor:** Prof. Dr. Thomas Kauê Dal'Maso Peron

---

## Overview

This repository provides a general-purpose framework for spatiotemporal crime network analysis.
Given any geolocated event dataset with timestamps, it:

1. Discretises space into a configurable grid of cells
2. Builds hourly binary event time series per cell
3. Measures pairwise similarity via **Event Synchronization (ES)**
4. Runs a Monte Carlo surrogate test to retain only statistically significant pairs
5. Constructs a complex network and extracts structural / local metrics and communities
6. Produces interactive Folium maps and Plotly time-series plots

The framework was applied to vehicle theft records from the state of São Paulo, Brazil (2022–2023), using data from the SSP-SP transparency portal. The thesis results are reproduced by the parameter set documented in the *[Reproducing the thesis](#reproducing-the-thesis)* section below.

---

## Pipeline

```
Raw event data (Excel / Parquet / CSV)
        │
        ▼
1. Preprocessing
   ├── Deduplication
   ├── String normalisation (diacritics, case)
   ├── Invalid coordinate removal
   └── Date filtering
        │
        ▼
2. Spatial grid  (configurable grid_size)
   └── Each event is assigned to a grid cell
        │
        ▼
3. Hourly event time series per cell
   └── Filter: keep only cells with >= min_events non-zero hours
        │
        ▼
4. Event Synchronization + Monte Carlo significance test
   [CPU-intensive — designed to run on an HPC cluster]
        │
        ▼
5. Adjacency matrix  (threshold: > n_surr * significance_level surrogates)
        │
        ▼
6. Complex network analysis
   ├── Structural: density, transitivity, assortativity, avg path length, diameter
   ├── Local: degree, betweenness, closeness, eigenvector centrality, clustering
   └── Communities: Girvan-Newman
        │
        ▼
7. Interactive visualisations (Folium maps, Plotly event series)
```

---

## Repository structure

```
.
├── src/
│   └── vehicle_theft_network/
│       ├── config.py            # centralised parameters (edit here to adapt)
│       ├── data/
│       │   ├── loader.py        # I/O helpers (Excel, Parquet, pickle, npy)
│       │   └── preprocessor.py  # cleaning & filtering pipeline
│       ├── grid/
│       │   └── builder.py       # spatial grid + cell aggregation
│       ├── timeseries/
│       │   └── builder.py       # time series construction & filtering
│       ├── event_sync/
│       │   └── analyzer.py      # ES analysis & adjacency matrix
│       ├── network/
│       │   ├── builder.py       # NetworkX graph construction
│       │   └── metrics.py       # structural + local metrics, communities
│       └── visualization/
│           ├── map_utils.py     # Folium map helpers
│           └── plots.py         # Plotly event series comparison
├── notebooks/                   # guided analysis notebooks (read-only outputs)
│   ├── 01_preprocessing.ipynb
│   ├── 02_grid_and_timeseries.ipynb
│   └── 03_network_analysis.ipynb
├── scripts/
│   ├── run_pipeline.py          # CLI orchestrator (all stages)
│   └── test_pipeline.py         # smoke test with synthetic data
├── data/                        # Parquet datasets tracked via Git LFS
├── dados/                       # local-only raw files (Excel, CSV — gitignored)
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/vehicle-theft-network.git
cd vehicle-theft-network

# 2. Install uv (if not already installed)
pip install uv

# 3. Create the virtual environment and install all dependencies
uv sync
```

`pyunicorn` is installed directly from its GitHub repository; `git` must be on your PATH.

---

## Usage

### Full pipeline (stages 1–4)

```bash
uv run python scripts/run_pipeline.py
```

### Including the visualisation stage

```bash
uv run python scripts/run_pipeline.py --visualize
```

### Single stage

```bash
uv run python scripts/run_pipeline.py --stage preprocess
uv run python scripts/run_pipeline.py --stage grid
uv run python scripts/run_pipeline.py --stage timeseries
uv run python scripts/run_pipeline.py --stage network
uv run python scripts/run_pipeline.py --stage visualize
```

### Smoke test (synthetic data, no real data required)

```bash
uv run python scripts/test_pipeline.py
```

### Notebooks

```bash
uv run jupyter notebook
```

Open `notebooks/` for guided, annotated walkthroughs of each pipeline stage.
The notebooks contain pre-computed outputs from the original thesis run; they are
intended as analysis references, not as the place to run the pipeline.

---

## Data

The framework accepts any tabular event dataset with at minimum:
- **Latitude / Longitude** columns (decimal degrees)
- **Date / time** columns
- A unique event identifier for deduplication

### SSP-SP dataset (thesis application)

The pre-processed dataset is provided as Parquet files tracked via **Git LFS**
(see `.gitattributes`). After cloning, run `git lfs pull` to download them:

| File | Description | Size |
|---|---|---|
| `data/VeiculosSubtraidos.parquet` | Raw records (all 3 years) | 32 MB |
| `data/roubo_veiculo_selecionado.parquet` | After preprocessing pipeline | 6.5 MB |

To work from the original Excel source instead, download the *Roubo de Veículo*
spreadsheets from the [SSP-SP transparency portal](https://www.ssp.sp.gov.br/transparenciassp),
place the file at `dados/VeiculosSubtraidos_2024.xlsx`, and update
`DataConfig.raw_data_path` to point to it. The loader detects the format by extension.

> **Monte Carlo note:** `event_sync/analyzer.py::compute_significance` is
> computationally expensive (hours per 1 000 surrogates on a single machine for
> ~1 300 cells). The pre-computed result used in the thesis (`monteCarlosimulations.npy`)
> was produced on an HPC cluster. To skip this stage, place the `.npy` file at
> `config.DataConfig.monte_carlo_path`.

---

## Reproducing the thesis

The exact parameter set used in the thesis is already the default in `config.py`:

| Parameter | Value | Description |
|---|---|---|
| `grid_size` | `0.002°` | ~200 m × 200 m cells at SP's latitude |
| `min_events` | `10` | minimum non-zero hours to retain a cell |
| `taumax` | `5` | ES lag window (hours) |
| `n_surr` | `1 000` | Monte Carlo surrogates |
| `significance_level` | `0.95` | 95% confidence threshold |
| Date range | 2022-01-01 → 2023-12-31 | |

Running `uv run python scripts/run_pipeline.py` with the SSP-SP data and the
pre-computed `monteCarlosimulations.npy` reproduces all reported metrics.

### Key numbers

| Metric | Value |
|---|---|
| Raw records (2022–2024) | 549 995 |
| Deduplicated incidents | 341 941 |
| After coord / date filters | ~276 503 |
| Grid cells created | 68 384 |
| Cells retained (>=10 events/h) | 1 336 |
| Network nodes (main component) | 1 319 |
| Link density | 0.0208 |
| Transitivity | 0.0853 |
| Average clustering | 0.1046 |
| Network diameter | 6 |
| Hub cell | (-46.605°, -23.561°) |

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@monography{freitas2023anatomy,
  author    = {Freitas, Vinícius Teixeira de Carvalho},
  title     = {A anatomia dos roubos de veículo no Estado de São Paulo:
               uma análise da dinâmica criminal utilizando redes complexas},
  school    = {Instituto de Ciências Matemáticas e de Computação,
               Universidade de São Paulo},
  year      = {2023},
  address   = {São Carlos},
  type      = {Trabalho de Conclusão de Curso}
}
```

---

## License

[MIT](LICENSE)
