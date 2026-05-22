from typing import Any, Dict

import networkx as nx


def compute_structural_metrics(G: nx.Graph) -> Dict[str, Any]:
    """Compute graph-level structural metrics.

    Metrics
    -------
    - ``link_density``: fraction of possible edges that are present.
    - ``transitivity``: global clustering coefficient (closed triangles / all triplets).
    - ``assortativity``: degree assortativity coefficient.
    - ``global_clustering``: mean local clustering coefficient.
    - ``avg_shortest_path_length``: mean over connected components with > 1 node.
    - ``connected_components``: number of connected components.
    - ``avg_degree``: mean node degree.
    """
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot compute metrics on an empty graph.")

    components = list(nx.connected_components(G))
    path_lengths = [
        nx.average_shortest_path_length(G.subgraph(c))
        for c in components
        if len(c) > 1
    ]
    degrees = dict(G.degree())
    return {
        "link_density": nx.density(G),
        "transitivity": nx.transitivity(G),
        "assortativity": nx.degree_assortativity_coefficient(G),
        "global_clustering": nx.average_clustering(G),
        "avg_shortest_path_length": sum(path_lengths) / len(path_lengths) if path_lengths else 0.0,
        "connected_components": len(components),
        "avg_degree": sum(degrees.values()) / G.number_of_nodes(),
    }


def compute_local_metrics(G: nx.Graph) -> Dict[str, Dict]:
    """Compute per-node centrality and clustering metrics.

    Returns a dict of dicts, one entry per metric, each mapping node -> value.
    Eigenvector centrality falls back to 0.0 for all nodes if the power
    iteration fails to converge.
    """
    if G.number_of_nodes() == 0:
        raise ValueError("Cannot compute metrics on an empty graph.")

    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = dict.fromkeys(G.nodes(), 0.0)

    return {
        "degree_centrality": nx.degree_centrality(G),
        "closeness_centrality": nx.closeness_centrality(G),
        "betweenness_centrality": nx.betweenness_centrality(G),
        "eigenvector_centrality": eigenvector,
        "clustering": nx.clustering(G),
        "degree": dict(G.degree()),
    }


def detect_communities(G: nx.Graph) -> Dict[int, list]:
    """Detect communities using the Girvan-Newman algorithm (first cut).

    The first iteration of Girvan-Newman is used, which removes the edge with
    the highest betweenness centrality and splits the graph into two parts.
    For finer-grained partitioning, iterate further or use a modularity-based
    method (e.g. ``nx.algorithms.community.greedy_modularity_communities``).
    """
    gen = nx.algorithms.community.centrality.girvan_newman(G)
    return {community_id: list(community) for community_id, community in enumerate(next(gen))}
