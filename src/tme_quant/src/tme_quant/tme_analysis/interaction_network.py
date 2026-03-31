"""
Graph-based analysis of TME interaction networks.

Builds a NetworkX graph from a list of InteractionPair objects and
provides:
  - Centrality measures (degree, betweenness, closeness, eigenvector)
  - Community detection (Louvain / greedy modularity)
  - Critical interaction identification (bridge edges, hub nodes)
  - Network-level metrics (density, modularity, clustering coefficient)

NetworkX is an optional dependency:
  pip install networkx

Example
-------
>>> from tme_quant.tme_analysis.interaction_network import (
...     InteractionNetwork, build_interaction_graph, compute_network_metrics,
... )
>>> graph = build_interaction_graph(interaction_pairs)
>>> metrics = compute_network_metrics(graph)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import InteractionPair


# ---------------------------------------------------------------------------
# Optional NetworkX import
# ---------------------------------------------------------------------------

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False


def _require_networkx() -> None:
    if not _NX_AVAILABLE:
        raise ImportError(
            "networkx is required for interaction network analysis. "
            "Install with: pip install networkx"
        )


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_interaction_graph(
    pairs: List[InteractionPair],
    node_type_attr: bool = True,
    weight_by: str = 'distance',
) -> "nx.Graph":
    """
    Build an undirected weighted NetworkX graph from interaction pairs.

    Each unique source/target id becomes a node; each pair becomes an edge.
    Edge weight defaults to the inverse of distance (closer = stronger).

    Parameters
    ----------
    pairs:
        List of InteractionPair objects.
    node_type_attr:
        When True, set a ``node_type`` attribute on each node
        (``'cell'``, ``'fiber'``, ``'tumor_boundary'``, etc.)
    weight_by:
        ``'distance'`` (default) → weight = 1 / (1 + distance)
        ``'tacs'``    → weight derived from TACS score (1=TACS-3, 0.5=TACS-2, 0=TACS-1)
        ``'equal'``   → all edges weight = 1.0

    Returns
    -------
    nx.Graph
    """
    _require_networkx()

    G = nx.Graph()

    for pair in pairs:
        src  = str(pair.source_id)
        tgt  = str(pair.target_id)

        # Nodes
        if node_type_attr:
            if src not in G:
                G.add_node(src, node_type=pair.source_type)
            if tgt not in G:
                G.add_node(tgt, node_type=pair.target_type)

        # Edge weight
        if weight_by == 'distance':
            w = 1.0 / (1.0 + pair.distance) if pair.distance is not None else 1.0
        elif weight_by == 'tacs':
            tacs_map = {'TACS-3': 1.0, 'TACS-2': 0.5, 'TACS-1': 0.2}
            w = tacs_map.get(str(pair.interaction_type), 0.1)
        else:
            w = 1.0

        # Parallel edges: accumulate weight
        if G.has_edge(src, tgt):
            G[src][tgt]['weight'] += w
            G[src][tgt]['interaction_count'] = G[src][tgt].get('interaction_count', 1) + 1
        else:
            G.add_edge(src, tgt,
                       weight=w,
                       distance=pair.distance,
                       interaction_type=str(pair.interaction_type),
                       interaction_count=1)

    return G


# ---------------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------------

def compute_centrality(
    G: "nx.Graph",
    measures: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute node centrality measures on the graph.

    Parameters
    ----------
    G:
        NetworkX graph.
    measures:
        Subset of ``['degree', 'betweenness', 'closeness', 'eigenvector']``.
        Defaults to all four.

    Returns
    -------
    Dict mapping measure name → dict of {node_id: score}.
    """
    _require_networkx()
    if measures is None:
        measures = ['degree', 'betweenness', 'closeness', 'eigenvector']

    results: Dict[str, Dict[str, float]] = {}

    if 'degree' in measures:
        results['degree'] = dict(nx.degree_centrality(G))

    if 'betweenness' in measures:
        results['betweenness'] = nx.betweenness_centrality(G, weight='weight')

    if 'closeness' in measures:
        results['closeness'] = nx.closeness_centrality(G)

    if 'eigenvector' in measures:
        try:
            results['eigenvector'] = nx.eigenvector_centrality(
                G, weight='weight', max_iter=500
            )
        except nx.PowerIterationFailedConvergence:
            results['eigenvector'] = {n: 0.0 for n in G.nodes()}

    return results


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------

def detect_communities(
    G: "nx.Graph",
    method: str = 'greedy',
) -> Dict[str, int]:
    """
    Detect communities (modules) in the interaction graph.

    Parameters
    ----------
    G:
        NetworkX graph.
    method:
        ``'greedy'`` — Clauset-Newman-Moore greedy modularity maximisation
            (built into networkx, no extra dependencies).
        ``'louvain'`` — Louvain algorithm (requires ``python-louvain``).

    Returns
    -------
    Dict mapping node_id → community integer label.
    """
    _require_networkx()

    if method == 'louvain':
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G, weight='weight')
            return partition
        except ImportError:
            raise ImportError(
                "python-louvain is required for Louvain community detection. "
                "Install with: pip install python-louvain"
            )

    # Default: greedy modularity
    communities = nx.community.greedy_modularity_communities(G, weight='weight')
    partition: Dict[str, int] = {}
    for cid, community in enumerate(communities):
        for node in community:
            partition[node] = cid
    return partition


# ---------------------------------------------------------------------------
# Critical interaction identification
# ---------------------------------------------------------------------------

def find_critical_interactions(
    G: "nx.Graph",
    top_n_hubs: int = 10,
) -> Dict[str, Any]:
    """
    Identify structurally important nodes and edges.

    Returns
    -------
    dict with:
        hub_nodes        : top-N nodes by degree centrality
        bridge_edges     : edges whose removal would disconnect components
        high_betweenness : top-N nodes by betweenness centrality
    """
    _require_networkx()

    degree_cent = nx.degree_centrality(G)
    betw_cent   = nx.betweenness_centrality(G, weight='weight')

    hub_nodes = sorted(degree_cent, key=degree_cent.get, reverse=True)[:top_n_hubs]
    high_betw = sorted(betw_cent,   key=betw_cent.get,   reverse=True)[:top_n_hubs]

    # Bridge edges (removing them increases number of connected components)
    bridge_edges = list(nx.bridges(G)) if nx.is_connected(G) else []

    return {
        'hub_nodes':         hub_nodes,
        'high_betweenness':  high_betw,
        'bridge_edges':      bridge_edges,
        'hub_degree_scores': {n: degree_cent[n] for n in hub_nodes},
        'hub_betweenness_scores': {n: betw_cent[n] for n in high_betw},
    }


# ---------------------------------------------------------------------------
# Network-level metrics
# ---------------------------------------------------------------------------

def compute_network_metrics(G: "nx.Graph") -> Dict[str, Any]:
    """
    Compute summary statistics for the full interaction graph.

    Returns
    -------
    dict with:
        n_nodes, n_edges, density, is_connected,
        n_components, avg_clustering, avg_degree,
        modularity (greedy partition), avg_path_length (largest component).
    """
    _require_networkx()

    metrics: Dict[str, Any] = {
        'n_nodes':       G.number_of_nodes(),
        'n_edges':       G.number_of_edges(),
        'density':       nx.density(G),
        'is_connected':  nx.is_connected(G),
        'n_components':  nx.number_connected_components(G),
        'avg_clustering': nx.average_clustering(G, weight='weight'),
        'avg_degree':    float(np.mean([d for _, d in G.degree()])) if G.nodes() else 0.0,
    }

    # Modularity of greedy partition
    try:
        communities = list(
            nx.community.greedy_modularity_communities(G, weight='weight')
        )
        metrics['modularity'] = nx.community.modularity(G, communities, weight='weight')
    except Exception:
        metrics['modularity'] = float('nan')

    # Average shortest path on the largest connected component
    if G.number_of_nodes() > 1:
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        if nx.is_connected(subgraph) and len(subgraph) > 1:
            try:
                metrics['avg_path_length'] = nx.average_shortest_path_length(
                    subgraph, weight=None
                )
            except Exception:
                metrics['avg_path_length'] = float('nan')

    return metrics


# ---------------------------------------------------------------------------
# High-level convenience class
# ---------------------------------------------------------------------------

@dataclass
class InteractionNetworkAnalyzer:
    """
    Convenience wrapper that runs the full network analysis pipeline.

    Parameters
    ----------
    weight_by:
        Edge weighting strategy passed to ``build_interaction_graph``.
    community_method:
        Community detection algorithm (``'greedy'`` or ``'louvain'``).
    verbose:
        Print progress messages.
    """
    weight_by:        str  = 'distance'
    community_method: str  = 'greedy'
    verbose:          bool = False

    def analyze(
        self,
        pairs: List[InteractionPair],
        top_n_hubs: int = 10,
    ) -> Dict[str, Any]:
        """
        Run the full analysis and return all results in a single dict.

        Returns
        -------
        dict with keys:
            graph, network_metrics, centrality, communities,
            critical_interactions.
        """
        if self.verbose:
            print(f'Building graph from {len(pairs)} interaction pairs …')

        G = build_interaction_graph(pairs, weight_by=self.weight_by)

        if self.verbose:
            print(f'Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.')

        results: Dict[str, Any] = {
            'graph': G,
            'network_metrics': compute_network_metrics(G),
        }

        if G.number_of_nodes() > 0:
            results['centrality'] = compute_centrality(G)
            results['communities'] = detect_communities(G, method=self.community_method)
            results['critical_interactions'] = find_critical_interactions(
                G, top_n_hubs=top_n_hubs
            )

        if self.verbose:
            m = results['network_metrics']
            print(f"  density={m['density']:.3f}  "
                  f"components={m['n_components']}  "
                  f"modularity={m.get('modularity', float('nan')):.3f}")

        return results


__all__ = [
    'build_interaction_graph',
    'compute_centrality',
    'detect_communities',
    'find_critical_interactions',
    'compute_network_metrics',
    'InteractionNetworkAnalyzer',
]