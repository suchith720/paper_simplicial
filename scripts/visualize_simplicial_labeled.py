#!/usr/bin/env python3
"""Labeled plot: simplicial edges (blue) vs 2-simplices / triangles (orange fill)."""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import EXTRACTED_DIR  # noqa: E402

EDGE_COLOR = "#1d4ed8"
TRI_FACE = (0.98, 0.55, 0.18, 0.42)
TRI_EDGE = "#c2410c"
NODE_COLOR = "#0f172a"
LABEL_COLOR = "#111827"


def load_record(path: Path) -> dict:
    with path.open("rb") as fp:
        return pickle.load(fp)["record"]


def load_deduped_stacked(run_dir: Path) -> np.ndarray | None:
    sp, kp = run_dir / "subsample/stacked.npy", run_dir / "dedup/keep_mask.npy"
    if not sp.is_file() or not kp.is_file():
        return None
    return np.load(sp)[np.load(kp).astype(bool)]


def truncate(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def pick_subgraph_nodes(
    triangles: list[tuple[int, int, int]],
    *,
    max_nodes: int,
    seed_triangle: int | None,
) -> set[int]:
    if not triangles:
        return set()
    if seed_triangle is not None:
        t = triangles[seed_triangle % len(triangles)]
        nodes = set(t)
    else:
        tri_count: dict[int, int] = {}
        for a, b, c in triangles:
            for x in (a, b, c):
                tri_count[x] = tri_count.get(x, 0) + 1
        hub = max(tri_count, key=tri_count.get)
        nodes = {hub}
        for a, b, c in sorted(triangles, key=lambda t: -len(set(t) & nodes)):
            if hub in (a, b, c):
                nodes.update([a, b, c])
            if len(nodes) >= max_nodes:
                break
    if len(nodes) < max_nodes:
        for a, b, c in triangles:
            if nodes & {a, b, c}:
                nodes.update([a, b, c])
            if len(nodes) >= max_nodes:
                break
    return set(sorted(nodes)[:max_nodes])


def layout_nodes(nodes: set[int], edges: list[tuple[int, int]], stacked: np.ndarray | None) -> dict[int, np.ndarray]:
    sub = [(u, v) for u, v in edges if u in nodes and v in nodes]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(sub)
    idx = sorted(nodes)
    if stacked is not None and len(idx) >= 2:
        X = stacked[list(idx)].astype(np.float64)
        X -= X.mean(axis=0)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        xy = X @ vt[:2].T
        return {idx[i]: xy[i] for i in range(len(idx))}
    return {n: np.array(p) for n, p in nx.spring_layout(G, seed=42, k=2.0 / max(len(nodes), 1)).items()}


def plot_labeled(
    record: dict,
    nodes: set[int],
    stacked: np.ndarray | None,
    out_path: Path,
    *,
    method: str,
    label_chars: int,
    font_size: float,
) -> None:
    edges_all = [tuple(sorted(e)) for e in record["edges"]]
    edge_w = {tuple(sorted(e)): float(w) for e, w in zip(record["edges"], record["edge_weights"])}
    tris_all = [tuple(sorted(t)) for t in record["triangles"]]
    titles = list(record["title"])

    sub_edges = [(u, v) for u, v in edges_all if u in nodes and v in nodes]
    sub_tris = [t for t in tris_all if all(x in nodes for x in t)]
    pos = layout_nodes(nodes, sub_edges, stacked)

    fig, ax = plt.subplots(figsize=(18, 14), facecolor="#f8fafc")
    ax.set_facecolor("#f8fafc")

    # 2-simplices (triangles) — orange layer
    if sub_tris:
        polys = [np.array([pos[a], pos[b], pos[c]]) for a, b, c in sub_tris]
        ax.add_collection(
            PolyCollection(
                polys,
                facecolors=[TRI_FACE],
                edgecolors=TRI_EDGE,
                linewidths=1.4,
                zorder=1,
            )
        )

    # 1-simplices (edges) — blue layer
    if sub_edges:
        segs = [np.array([pos[u], pos[v]]) for u, v in sub_edges]
        w = np.array([edge_w.get(e, 0.0) for e in sub_edges])
        wn = (w - w.min()) / (np.ptp(w) + 1e-8)
        ax.add_collection(
            LineCollection(
                segs,
                colors=EDGE_COLOR,
                linewidths=1.0 + 2.5 * wn,
                alpha=0.75 + 0.2 * wn,
                zorder=2,
            )
        )

    # Nodes + paper title labels
    for n in sorted(nodes):
        p = pos[n]
        ax.scatter(
            p[0], p[1], s=220, c=NODE_COLOR, edgecolors="white", linewidths=2, zorder=4
        )
        ax.text(
            p[0],
            p[1],
            truncate(titles[n], label_chars),
            fontsize=font_size,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=LABEL_COLOR, edgecolor="#64748b", alpha=0.95),
        )

    ax.set_title(
        f"Simplicial complex ({method}) — "
        f"{len(nodes)} papers, {len(sub_edges)} edges, {len(sub_tris)} triangles",
        fontsize=14,
        pad=14,
    )
    ax.axis("off")
    ax.autoscale()
    ax.margins(0.15)

    legend = [
        mpatches.Patch(facecolor=EDGE_COLOR, edgecolor=EDGE_COLOR, label="1-simplices (kNN edges)"),
        mpatches.Patch(facecolor=TRI_FACE, edgecolor=TRI_EDGE, label="2-simplices (triangles)"),
        mpatches.Patch(facecolor=NODE_COLOR, edgecolor="white", label="Papers (node labels = title)"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=10, framealpha=0.95)

    fig.text(
        0.5,
        0.01,
        "Orange regions: three mutually connected papers. Blue lines: pairwise similarity edges from the complex.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=EXTRACTED_DIR / "run_10k")
    p.add_argument("--method", choices=("clique", "neighbor"), default="clique")
    p.add_argument("--max-nodes", type=int, default=24)
    p.add_argument("--seed-triangle", type=int, default=None)
    p.add_argument("--label-chars", type=int, default=55)
    p.add_argument("--font-size", type=float, default=6.5)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    rec_path = args.run_dir / "complexes" / args.method / "complex.pkl"
    if not rec_path.is_file():
        print(f"Missing {rec_path}", file=sys.stderr)
        return 1

    record = load_record(rec_path)
    triangles = [tuple(sorted(t)) for t in record["triangles"]]
    nodes = pick_subgraph_nodes(triangles, max_nodes=args.max_nodes, seed_triangle=args.seed_triangle)
    if not nodes:
        print("No triangles in complex; cannot pick a labeled simplicial patch.", file=sys.stderr)
        return 1

    stacked = load_deduped_stacked(args.run_dir)
    out = args.out or (args.run_dir / "figures" / f"{args.method}_labeled_edges_triangles.png")
    plot_labeled(
        record,
        nodes,
        stacked,
        out,
        method=args.method,
        label_chars=args.label_chars,
        font_size=args.font_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
