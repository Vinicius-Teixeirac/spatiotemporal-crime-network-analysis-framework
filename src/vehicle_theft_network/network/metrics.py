from typing import Any, Dict

import networkx as nx


def compute_structural_metrics(G: nx.Graph) -> Dict[str, Any]:
    components = list(nx.connected_components(G))
    path_lengths = []
    for c in components:
        sub = G.subgraph(c)
        if sub.number_of_nodes() > 1:
            path_lengths.append(nx.average_shortest_path_length(sub))
    avg_path_length = sum(path_lengths) / len(path_lengths) if path_lengths else 0.0
    degrees = dict(G.degree())
    return {
        "link_density": nx.density(G),
        "transitivity": nx.transitivity(G),
        "assortativity": nx.degree_assortativity_coefficient(G),
        "global_clustering": nx.average_clustering(G),
        "avg_shortest_path_length": avg_path_length,
        "connected_components": len(components),
        "avg_degree": sum(degrees.values()) / G.number_of_nodes(),
    }


def compute_local_metrics(G: nx.Graph) -> Dict[str, Dict]:
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
    gen = nx.algorithms.community.centrality.girvan_newman(G)
    community_nodes: Dict[int, list] = {}
    for community_id, community in enumerate(next(gen)):
        community_nodes[community_id] = list(community)
    return community_nodes
