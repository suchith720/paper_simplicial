"""Build an undirected kNN graph from paper embedding vectors."""
from __future__ import annotations

import networkx as nx
import numpy as np


def center_features(x: np.ndarray, mean: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    if mean is None:
        mean = x.mean(axis=0)
    return x - mean, mean


def knn_undirected_graph(
    x: np.ndarray,
    k: int,
    *,
    mean: np.ndarray | None = None,
    query_block: int = 4096,
) -> tuple[nx.Graph, np.ndarray, np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as F

    xc, mean = center_features(x, mean)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.from_numpy(xc).to(device)
    t = F.normalize(t, dim=1)

    n = t.shape[0]
    k_eff = min(int(k), max(n - 1, 0))
    if k_eff < 1:
        raise ValueError("Need n >= 2 and k >= 1 for kNN graph.")

    knn_idx = np.empty((n, k_eff), dtype=np.int64)
    knn_sim = np.empty((n, k_eff), dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, n, query_block):
            e = min(s + query_block, n)
            sim = t[s:e] @ t.T
            sim[torch.arange(e - s, device=device), torch.arange(s, e, device=device)] = -torch.inf
            v, idx = sim.topk(k_eff, dim=1)
            knn_idx[s:e] = idx.cpu().numpy()
            knn_sim[s:e] = v.float().cpu().numpy()

    G = nx.Graph()
    G.add_nodes_from(range(n))
    src = np.repeat(np.arange(n), k_eff)
    dst = knn_idx.ravel()
    weight = knn_sim.ravel()
    a, b = np.minimum(src, dst), np.maximum(src, dst)
    _, first = np.unique(a * n + b, return_index=True)
    for u, v, w in zip(a[first], b[first], weight[first]):
        if u != v:
            G.add_edge(int(u), int(v), weight=float(w))
    return G, knn_idx, knn_sim, mean
