from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd


def get_coord_labels(time_series_df: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
    """Map integer node indices to ``(lon, lat)`` coordinate tuples.

    The mapping is derived from the column order of ``time_series_df``, so
    node ``i`` corresponds to the i-th column (i.e. the i-th retained cell).
    """
    return {i: col for i, col in enumerate(time_series_df.columns)}


def build_network(
    adjacency_matrix: np.ndarray,
    coord_labels: Dict[int, Tuple[float, float]],
    directed: bool = False,
) -> nx.Graph:
    """Construct a NetworkX graph from a binary adjacency matrix.

    Node attributes are not set here; use ``coord_labels`` to map node
    indices back to spatial coordinates downstream.

    Parameters
    ----------
    adjacency_matrix:
        Square binary matrix of shape ``(n_cells, n_cells)``.
    coord_labels:
        Integer-to-coordinate mapping returned by :func:`get_coord_labels`.
    directed:
        If ``True``, returns a ``DiGraph``; otherwise an undirected ``Graph``.
    """
    cls = nx.DiGraph if directed else nx.Graph
    return nx.from_numpy_array(adjacency_matrix, create_using=cls)
