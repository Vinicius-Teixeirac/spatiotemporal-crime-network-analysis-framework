import numpy as np
from pyunicorn.eventseries import EventSeries as EV


def build_event_series(event_array: np.ndarray, taumax: int = 5) -> EV:
    return EV(event_array, taumax=taumax)


def compute_significance(
    ev: EV,
    n_surr: int,
    symmetrization: str = "symmetric",
) -> np.ndarray:
    """Run Monte Carlo surrogate test.

    Returns a matrix where entry (i, j) counts how many surrogates showed
    synchronization between cells i and j. Intended to be run on a cluster
    due to the high computational cost.
    """
    n_cells = ev.get_event_matrix().shape[1]
    counts = np.zeros((n_cells, n_cells), dtype=np.int16)
    for _ in range(n_surr):
        result = ev.event_analysis_significance(
            method="ES", n_surr=1, symmetrization=symmetrization
        )
        counts += result.round().astype(np.int16)
    return counts


def build_adjacency_matrix(
    monte_carlo_counts: np.ndarray,
    n_surr: int,
    significance_level: float = 0.95,
) -> np.ndarray:
    threshold = int(n_surr * significance_level)
    return np.where(monte_carlo_counts > threshold, 1, 0)
