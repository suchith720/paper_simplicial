#!/usr/bin/env python3
"""Triangle-only simplicial view: large color-coded nodes, no edge lines, no labels."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from matplotlib.collections import PolyCollection
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import EXTRACTED_DIR  # noqa: E402

TRI_FACE = (1.0, 0.72, 0.38, 0.55)
TRI_EDGE = "#ea580c"
PALETTE = plt.cm.tab20.colors


def load_record(path: Path) -> dict:
    with path.open("rb") as fp:
        return pickle.load(fp)["record"]


def load_deduped_stacked(run_dir: Path) -> np.ndarray:
    full = np.load(run_dir / "subsample/stacked.npy")
    keep = np.load(run_dir / "dedup/keep_mask.npy").astype(bool)
    return full[keep]


def cluster_labels(stacked: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    X = stacked.astype(np.float64)
    X -= X.mean(axis=0)
    n = X.shape[0]
    k = min(int(n_clusters), max(n, 1))
    if k <= 1:
        return np.zeros(n, dtype=np.int32)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return km.fit_predict(X).astype(np.int32)


def pick_triangle_patch(
    triangles: list[tuple[int, int, int]],
    *,
    max_triangles: int,
    seed: int,
) -> tuple[set[int], list[tuple[int, int, int]]]:
    if not triangles:
        return set(), []
    rng = np.random.default_rng(seed)
    start = triangles[int(rng.integers(0, len(triangles)))]
    chosen_tris = [start]
    nodes = set(start)
    for t in triangles:
        if set(t) & nodes:
            if t not in chosen_tris:
                chosen_tris.append(t)
            nodes.update(t)
        if len(chosen_tris) >= max_triangles:
            break
    if len(chosen_tris) < max_triangles:
        for t in triangles:
            if t not in chosen_tris:
                chosen_tris.append(t)
                nodes.update(t)
            if len(chosen_tris) >= max_triangles:
                break
    return nodes, chosen_tris[:max_triangles]


def layout_patch(nodes: set[int], tris: list[tuple[int, int, int]], stacked: np.ndarray) -> dict[int, np.ndarray]:
    idx = sorted(nodes)
    pos_map = {i: k for k, i in enumerate(idx)}
    G = nx.Graph()
    G.add_nodes_from(idx)
    for a, b, c in tris:
        G.add_edge(a, b)
        G.add_edge(b, c)
        G.add_edge(a, c)
    X = stacked[list(idx)].astype(np.float64)
    X -= X.mean(axis=0)
    if X.shape[0] >= 3:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        xy = X @ vt[:2].T
        return {idx[i]: xy[i] for i in range(len(idx))}
    return {n: np.array(p) for n, p in nx.spring_layout(G, seed=42).items()}


def plot_triangle_patch(
    *,
    tris: list[tuple[int, int, int]],
    nodes: set[int],
    pos: dict[int, np.ndarray],
    node_colors: np.ndarray,
    out_path: Path,
    method: str,
    n_clusters: int,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 11), facecolor="white")
    ax.set_facecolor("#fafafa")

    polys = [np.array([pos[a], pos[b], pos[c]]) for a, b, c in tris]
    ax.add_collection(
        PolyCollection(
            polys,
            facecolors=[TRI_FACE],
            edgecolors=TRI_EDGE,
            linewidths=2.0,
            zorder=1,
        )
    )

    idx = sorted(nodes)
    xy = np.array([pos[n] for n in idx])
    colors = [PALETTE[int(node_colors[n]) % len(PALETTE)] for n in idx]
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=520,
        c=colors,
        edgecolors="#1e293b",
        linewidths=2.5,
        zorder=3,
    )

    ax.set_title(
        f"{method.title()} complex — {len(tris)} highlighted triangles ({len(nodes)} papers)\n"
        f"Node colors = embedding clusters (k={n_clusters}); edges omitted (given by triangles)",
        fontsize=13,
        pad=12,
    )
    ax.axis("off")
    ax.autoscale()
    ax.margins(0.18)

    cluster_patches = [
        mpatches.Patch(facecolor=PALETTE[i % len(PALETTE)], edgecolor="#1e293b", label=f"cluster {i}")
        for i in range(min(n_clusters, 10))
    ]
    if n_clusters > 10:
        cluster_patches.append(mpatches.Patch(facecolor="#ccc", label=f"… +{n_clusters - 10} more"))
    tri_patch = mpatches.Patch(facecolor=TRI_FACE, edgecolor=TRI_EDGE, label="2-simplex (triangle)")
    ax.legend(
        handles=[tri_patch] + cluster_patches,
        loc="upper left",
        fontsize=9,
        framealpha=0.95,
        title="Legend",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=EXTRACTED_DIR / "run_10k")
    p.add_argument("--method", choices=("clique", "neighbor"), default="clique")
    p.add_argument("--max-triangles", type=int, default=18)
    p.add_argument("--n-clusters", type=int, default=12, help="KMeans on concat embeddings (no dataset labels)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    rec_path = args.run_dir / "complexes" / args.method / "complex.pkl"
    if not rec_path.is_file():
        print(f"Missing {rec_path}", file=sys.stderr)
        return 1

    record = load_record(rec_path)
    triangles = [tuple(sorted(t)) for t in record["triangles"]]
    stacked = load_deduped_stacked(args.run_dir)
    labels = cluster_labels(stacked, args.n_clusters, args.seed)

    nodes, tris = pick_triangle_patch(triangles, max_triangles=args.max_triangles, seed=args.seed)
    if not tris:
        print("No triangles to plot.", file=sys.stderr)
        return 1

    pos = layout_patch(nodes, tris, stacked)
    out = args.out or (
        args.run_dir / "figures" / f"{args.method}_triangles_only_clustered.png"
    )
    plot_triangle_patch(
        tris=tris,
        nodes=nodes,
        pos=pos,
        node_colors=labels,
        out_path=out,
        method=args.method,
        n_clusters=min(args.n_clusters, len(stacked)),
    )

    meta = {
        "dataset_has_official_classes": False,
        "node_coloring": f"KMeans k={args.n_clusters} on concat(head, windows_mean)",
        "n_triangles_shown": len(tris),
        "n_nodes_shown": len(nodes),
        "method": args.method,
        "output": str(out),
    }
    meta_path = args.run_dir / "figures" / f"{args.method}_triangles_viz_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
