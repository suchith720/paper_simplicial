#!/usr/bin/env python3
"""All papers or a random subset: edges + highlighted triangles, large nodes, no labels."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import EXTRACTED_DIR  # noqa: E402

EDGE_COLOR = "#94a3b8"
TRI_FACE = (0.93, 0.35, 0.42, 0.38)
TRI_EDGE = "#be123c"
NODE_COLOR = "#0f172a"
TRI_BATCH = 6000


def load_record(path: Path) -> dict:
    with path.open("rb") as fp:
        return pickle.load(fp)["record"]


def layout_pca2(stacked: np.ndarray) -> np.ndarray:
    X = stacked.astype(np.float64)
    X -= X.mean(axis=0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return X @ vt[:2].T


def induce_on_nodes(
    record: dict,
    node_ids: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]], int]:
    """Induced subcomplex; remapped indices 0 .. len(node_ids)-1."""
    keep = set(int(i) for i in node_ids)
    old_to_new = {int(o): n for n, o in enumerate(node_ids)}
    edges: list[tuple[int, int]] = []
    for e in record["edges"]:
        u, v = int(e[0]), int(e[1])
        if u in keep and v in keep:
            a, b = old_to_new[u], old_to_new[v]
            if a != b:
                edges.append(tuple(sorted((a, b))))
    tris: list[tuple[int, int, int]] = []
    for t in record["triangles"]:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        if a in keep and b in keep and c in keep:
            tris.append(
                tuple(sorted((old_to_new[a], old_to_new[b], old_to_new[c])))
            )
    return edges, tris, len(node_ids)


def plot_mesh(
    pos: np.ndarray,
    edges: list[tuple[int, int]],
    triangles: list[tuple[int, int, int]],
    out_path: Path,
    *,
    method: str,
    node_size: float,
    edge_alpha: float,
    edge_lw: float,
    tri_alpha: float,
    title_suffix: str,
    caption: str,
    figsize: tuple[float, float],
) -> None:
    n = pos.shape[0]
    fig, ax = plt.subplots(figsize=figsize, facecolor="#ffffff")
    ax.set_facecolor("#f1f5f9")

    if edges:
        segs = np.array([[pos[u], pos[v]] for u, v in edges], dtype=np.float64)
        ax.add_collection(
            LineCollection(
                segs,
                colors=EDGE_COLOR,
                linewidths=edge_lw,
                alpha=edge_alpha,
                zorder=1,
            )
        )

    for start in range(0, len(triangles), TRI_BATCH):
        batch = triangles[start : start + TRI_BATCH]
        polys = np.array([[pos[a], pos[b], pos[c]] for a, b, c in batch], dtype=np.float64)
        ax.add_collection(
            PolyCollection(
                polys,
                facecolors=[TRI_FACE],
                edgecolors=TRI_EDGE,
                linewidths=0.25,
                alpha=tri_alpha,
                zorder=2,
            )
        )

    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        s=node_size,
        c=NODE_COLOR,
        alpha=0.92,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )

    ax.set_title(
        f"{title_suffix} — {method} complex\n"
        f"{n:,} papers  ·  {len(edges):,} edges  ·  {len(triangles):,} triangles",
        fontsize=14,
        pad=10,
    )
    ax.axis("off")
    ax.autoscale()
    ax.margins(0.04)

    legend = [
        mpatches.Patch(facecolor=EDGE_COLOR, edgecolor=EDGE_COLOR, alpha=0.5, label="Edges (1-simplices)"),
        mpatches.Patch(facecolor=TRI_FACE, edgecolor=TRI_EDGE, label="Triangles (2-simplices)"),
        mpatches.Patch(facecolor=NODE_COLOR, edgecolor="white", label="Papers (nodes)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=10, framealpha=0.95)
    fig.text(0.5, 0.01, caption, ha="center", fontsize=9, color="#64748b")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_path}  ({n} nodes, {len(edges)} edges, {len(triangles)} triangles)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=EXTRACTED_DIR / "run_10k")
    p.add_argument("--method", choices=("clique", "neighbor"), default="clique")
    p.add_argument("--sample-size", type=int, default=0, help="0 = all papers; e.g. 2000 for random subset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--node-size", type=float, default=28)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    rec_path = args.run_dir / "complexes" / args.method / "complex.pkl"
    if not rec_path.is_file():
        print(f"Missing {rec_path}", file=sys.stderr)
        return 1

    record = load_record(rec_path)
    full = np.load(args.run_dir / "subsample/stacked.npy")
    keep = np.load(args.run_dir / "dedup/keep_mask.npy").astype(bool)
    stacked = full[keep]
    n_all = record["n_nodes"]

    if args.sample_size and args.sample_size < n_all:
        rng = np.random.default_rng(args.seed)
        node_ids = np.sort(rng.choice(n_all, size=args.sample_size, replace=False))
        edges, tris, n = induce_on_nodes(record, node_ids)
        pos = layout_pca2(stacked[node_ids])
        tag = f"{args.sample_size}samples"
        out = args.out or (args.run_dir / "figures" / f"{args.method}_mesh_{tag}.png")
        sample_path = args.run_dir / "figures" / f"{args.method}_mesh_{tag}_node_ids.npy"
        np.save(sample_path, node_ids)
        title_suffix = f"Random {args.sample_size:,} papers (induced subcomplex)"
        caption = (
            f"Layout: PCA on sampled concat embeddings. "
            f"Induced edges/triangles from saved {args.method} complex (seed={args.seed}). "
            f"Node ids: {sample_path.name}"
        )
        plot_mesh(
            pos,
            edges,
            tris,
            out,
            method=args.method,
            node_size=max(args.node_size, 40),
            edge_alpha=0.35,
            edge_lw=0.45,
            tri_alpha=0.62,
            title_suffix=title_suffix,
            caption=caption,
            figsize=(18, 14),
        )
        meta = {
            "sample_size": int(args.sample_size),
            "seed": args.seed,
            "node_ids_file": str(sample_path),
            "n_nodes": n,
            "n_edges": len(edges),
            "n_triangles": len(tris),
            "method": args.method,
            "output": str(out),
        }
        meta_path = args.run_dir / "figures" / f"{args.method}_mesh_{tag}_meta.json"
    else:
        edges = [tuple(sorted((int(e[0]), int(e[1])))) for e in record["edges"]]
        tris = [tuple(sorted((int(t[0]), int(t[1]), int(t[2])))) for t in record["triangles"]]
        pos = layout_pca2(stacked)
        tag = "all_papers"
        out = args.out or (args.run_dir / "figures" / f"{args.method}_full_mesh_{tag}.png")
        title_suffix = "Full deduped chunk"
        caption = "Layout: PCA of concat embeddings. Whole deduped chunk used for extraction."
        plot_mesh(
            pos,
            edges,
            tris,
            out,
            method=args.method,
            node_size=args.node_size,
            edge_alpha=0.22,
            edge_lw=0.25,
            tri_alpha=0.55,
            title_suffix=title_suffix,
            caption=caption,
            figsize=(20, 16),
        )
        meta = {
            "sample_size": n_all,
            "n_nodes": n_all,
            "n_edges": len(edges),
            "n_triangles": len(tris),
            "method": args.method,
            "output": str(out),
        }
        meta_path = args.run_dir / "figures" / f"{args.method}_full_mesh_meta.json"

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
