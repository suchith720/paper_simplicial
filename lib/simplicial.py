"""TopoNetX lifts from a paper kNN graph."""
from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np
from toponetx.transform.graph_to_simplicial_complex import (
    graph_to_clique_complex,
    graph_to_neighbor_complex,
    weighted_graph_to_vietoris_rips_complex,
)

Method = Literal["clique", "neighbor", "vietoris_rips"]


def _simplex_vertices(simplex) -> tuple[int, ...]:
    if hasattr(simplex, "elements"):
        return tuple(sorted(int(v) for v in simplex.elements))
    return tuple(sorted(int(v) for v in simplex))


def skeleton_edges_triangles(sc, max_rank: int) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]]:
    edges: list[tuple[int, int]] = []
    triangles: list[tuple[int, int, int]] = []
    if sc.dim >= 1:
        for s in sc.skeleton(1):
            u, v = _simplex_vertices(s)
            edges.append((u, v))
    if sc.dim >= 2 and max_rank >= 2:
        for s in sc.skeleton(2):
            a, b, c = _simplex_vertices(s)
            triangles.append((a, b, c))
    return edges, triangles


def neighbor_rank2_skeleton(G: nx.Graph) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]]:
    """Rank-2 neighbor skeleton: graph edges + common-neighbor triangles (scalable)."""
    edges = [tuple(sorted((int(u), int(v)))) for u, v in G.edges()]
    triangles: set[tuple[int, int, int]] = set()
    for u, v in G.edges():
        nu = set(G.neighbors(u))
        for w in G.neighbors(v):
            if w != u and w in nu:
                triangles.add(tuple(sorted((int(u), int(v), int(w)))))
    return edges, sorted(triangles)


def distance_graph_from_knn(G: nx.Graph) -> nx.Graph:
    """kNN edges only, weight = 1 - cosine (avoids O(n^2) complete graph)."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 0.0))
        H.add_edge(u, v, weight=float(1.0 - w))
    return H


def lift_graph(
    G: nx.Graph,
    method: Method,
    *,
    max_rank: int = 2,
    vr_radius: float | None = None,
    max_dim: int = 2,
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]], float | None]:
    if method == "clique":
        Gu = nx.Graph(G)
        for u, v in Gu.edges():
            Gu[u][v].clear()
        sc = graph_to_clique_complex(Gu, max_rank=max_rank)
        edges, triangles = skeleton_edges_triangles(sc, max_rank)
        return edges, triangles, None

    if method == "neighbor":
        return (*neighbor_rank2_skeleton(G), None)

    if method == "vietoris_rips":
        H = distance_graph_from_knn(G)
        weights = [float(d.get("weight", 0.0)) for _, _, d in H.edges(data=True)]
        r = float(vr_radius) if vr_radius is not None else float(np.median(weights))
        sc = weighted_graph_to_vietoris_rips_complex(H, r=r, max_dim=max_dim)
        edges, triangles = skeleton_edges_triangles(sc, max_rank)
        return edges, triangles, r

    raise ValueError(f"Unknown method: {method!r}")
