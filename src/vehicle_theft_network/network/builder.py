from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd


def get_coord_labels(time_series_df: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
    return {i: col for i, col in enumerate(time_series_df.columns)}


def build_network(
    adjacency_matrix: np.ndarray,
    coord_labels: Dict[int, Tuple[float, float]],
    directed: bool = False,
) -> nx.Graph:
    cls = nx.DiGraph if directed else nx.Graph
    return nx.from_numpy_array(adjacency_matrix, create_using=cls)
