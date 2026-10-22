"""Merge near-duplicate papers by cosine similarity on stacked embeddings."""
from __future__ import annotations

import numpy as np


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return x / n


def merge_clusters_at_threshold(
    x: np.ndarray,
    threshold: float,
    *,
    block: int = 2000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    n = x.shape[0]
    xn = l2_normalize(x.astype(np.float64, copy=False))
    uf = UnionFind(n)
    thr = float(threshold)
    n_pairs = 0
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        for j0 in range(i0, n, block):
            j1 = min(n, j0 + block)
            sim = xn[i0:i1] @ xn[j0:j1].T
            for ii in range(i1 - i0):
                for jj in range(j1 - j0):
                    gi, gj = i0 + ii, j0 + jj
                    if gi >= gj:
                        continue
                    if float(sim[ii, jj]) >= thr:
                        uf.union(gi, gj)
                        n_pairs += 1

    roots = np.asarray([uf.find(i) for i in range(n)], dtype=np.int64)
    rep_for_root: dict[int, int] = {}
    for i in range(n):
        r = int(roots[i])
        if r not in rep_for_root or i < rep_for_root[r]:
            rep_for_root[r] = i
    cluster_id = np.asarray([rep_for_root[int(roots[i])] for i in range(n)], dtype=np.int64)
    keep = np.zeros(n, dtype=bool)
    for rep in rep_for_root.values():
        keep[rep] = True

    sizes: dict[int, int] = {}
    for i in range(n):
        r = int(cluster_id[i])
        sizes[r] = sizes.get(r, 0) + 1
    multi = sorted((c for c in sizes.values() if c > 1), reverse=True)
    stats = {
        "n_papers_in": n,
        "n_papers_out": int(keep.sum()),
        "n_merged_away": n - int(keep.sum()),
        "n_pairs_at_threshold": n_pairs,
        "n_clusters_multi": sum(1 for c in sizes.values() if c > 1),
        "largest_cluster_sizes": multi[:10],
    }
    return cluster_id, keep, stats
