#!/usr/bin/env python3
"""Visualize saved simplicial complexes: full chunk or local labeled subgraph."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import EXTRACTED_DIR  # noqa: E402


def load_record(path: Path) -> dict:
    with path.open("rb") as fp:
        obj = pickle.load(fp)
    return obj["record"]


def load_deduped_stacked(run_dir: Path) -> np.ndarray | None:
    stacked_path = run_dir / "subsample" / "stacked.npy"
    keep_path = run_dir / "dedup" / "keep_mask.npy"
    if not stacked_path.is_file() or not keep_path.is_file():
        return None
    full = np.load(stacked_path)
    keep = np.load(keep_path).astype(bool)
    return full[keep]


def layout_pca2(stacked: np.ndarray) -> np.ndarray:
    """(n, 2) layout from centered stacked embeddings."""
    X = stacked.astype(np.float64)
    X = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return X @ vt[:2].T


def triangle_counts(triangles: list[tuple[int, int, int]], n_nodes: int) -> np.ndarray:
    counts = np.zeros(n_nodes, dtype=np.int32)
    for a, b, c in triangles:
        counts[a] += 1
        counts[b] += 1
        counts[c] += 1
    return counts


def select_triangle_dense_nodes(
    triangles: list[tuple[int, int, int]],
    edges: set[tuple[int, int]],
    *,
    max_nodes: int,
    seed: int | None,
) -> set[int]:
    if not triangles:
        nodes: set[int] = set()
        for u, v in edges:
            nodes.add(u)
            nodes.add(v)
            if len(nodes) >= max_nodes:
                break
        return nodes

    tri_count: dict[int, int] = {}
    for a, b, c in triangles:
        for x in (a, b, c):
            tri_count[x] = tri_count.get(x, 0) + 1

    start = seed if seed is not None else max(tri_count, key=tri_count.get)
    chosen = {start}
    tris = sorted(triangles, key=lambda t: sum(tri_count.get(x, 0) for x in t), reverse=True)
    for a, b, c in tris:
        if start in (a, b, c) or chosen & {a, b, c}:
            chosen.update([a, b, c])
        if len(chosen) >= max_nodes:
            break
    if len(chosen) < max_nodes:
        for a, b, c in tris:
            chosen.update([a, b, c])
            if len(chosen) >= max_nodes:
                break
    return set(sorted(chosen)[:max_nodes])


def layout_positions_local(
    nodes: set[int],
    edges: list[tuple[int, int]],
    stacked: np.ndarray | None,
) -> dict[int, np.ndarray]:
    sub_edges = [(u, v) for u, v in edges if u in nodes and v in nodes]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(sub_edges)
    if stacked is not None and len(nodes) >= 3:
        idx = sorted(nodes)
        X = stacked[list(idx)].astype(np.float64)
        X = X - X.mean(axis=0)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        coords = X @ vt[:2].T
        return {idx[i]: coords[i] for i in range(len(idx))}
    return {n: np.array(v) for n, v in nx.spring_layout(G, seed=0, k=1.5 / max(len(nodes), 1)).items()}


def truncate(s: str, n: int = 48) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def plot_full_chunk(
    record: dict,
    stacked: np.ndarray,
    out_png: Path,
    out_html: Path | None,
    *,
    label_top: int,
    edge_alpha: float,
    title: str,
) -> None:
    n = record["n_nodes"]
    edges = [tuple(sorted(e)) for e in record["edges"]]
    edge_w = {tuple(sorted(e)): w for e, w in zip(record["edges"], record["edge_weights"])}
    triangles = [tuple(sorted(t)) for t in record["triangles"]]
    titles = list(record["title"])
    paper_ids = list(record["paper_id"])

    pos = layout_pca2(stacked)
    tri_n = triangle_counts(triangles, n)

    fig, axes = plt.subplots(1, 2, figsize=(22, 10), facecolor="#fafafa")
    ax_field, ax_graph = axes
    for ax in axes:
        ax.set_facecolor("#0d1117")

    # --- Left: triangle density field (all 2-simplices) ---
    if triangles:
        tri_arr = np.asarray(triangles, dtype=np.int64)
        cent = pos[tri_arr].mean(axis=1)
        ax_field.hexbin(
            cent[:, 0],
            cent[:, 1],
            gridsize=90,
            cmap="inferno",
            mincnt=1,
            linewidths=0,
            alpha=0.95,
        )
    ax_field.set_title("Higher-order density (all triangle centroids)", color="white", fontsize=12)
    ax_field.axis("off")

    # --- Right: full 1-skeleton + nodes ---
    segs = [[pos[u], pos[v]] for u, v in edges]
    if segs:
        weights = np.array([edge_w.get(e, 0.0) for e in edges], dtype=np.float32)
        w_norm = np.clip((weights - weights.min()) / (np.ptp(weights) + 1e-8), 0, 1)
        lc = LineCollection(
            segs,
            colors=plt.cm.plasma(0.15 + 0.55 * w_norm),
            linewidths=0.12,
            alpha=edge_alpha,
            zorder=1,
        )
        ax_graph.add_collection(lc)

    # Faint triangle fills (batched, whole complex)
    if triangles:
        batch = 8000
        for start in range(0, len(triangles), batch):
            batch_tris = triangles[start : start + batch]
            polys = [pos[[a, b, c]] for a, b, c in batch_tris]
            pc = PolyCollection(
                polys,
                facecolors=(1.0, 0.45, 0.35, 0.012),
                edgecolors="none",
                zorder=0,
            )
            ax_graph.add_collection(pc)

    log_tri = np.log1p(tri_n.astype(np.float64))
    sc = ax_graph.scatter(
        pos[:, 0],
        pos[:, 1],
        c=log_tri,
        s=4 + 12 * (log_tri / (log_tri.max() + 1e-8)),
        cmap="viridis",
        alpha=0.9,
        edgecolors="none",
        zorder=2,
    )
    cbar = fig.colorbar(sc, ax=ax_graph, fraction=0.035, pad=0.02)
    cbar.set_label("log(1 + triangle count per paper)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    if label_top > 0:
        top_idx = np.argsort(-tri_n)[:label_top]
        for i in top_idx:
            ax_graph.text(
                pos[i, 0],
                pos[i, 1],
                truncate(titles[i], 42),
                fontsize=5,
                color="white",
                alpha=0.92,
                ha="center",
                va="bottom",
                zorder=3,
            )

    ax_graph.set_title("Full chunk: all papers, edges & 2-simplices", color="white", fontsize=12)
    ax_graph.axis("off")
    ax_graph.autoscale()

    fig.suptitle(title, fontsize=14, color="#eaeaea", y=0.98)
    caption = (
        f"Whole deduped chunk: {n:,} papers  |  {len(edges):,} simplicial edges  |  "
        f"{len(triangles):,} triangles. Layout: PCA of concat embeddings. "
        f"Hover titles: {out_html.name if out_html else 'N/A'}."
    )
    fig.text(0.5, 0.01, caption, ha="center", fontsize=9, color="#888888")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_png}")

    if out_html is not None:
        write_plotly_full(pos, titles, paper_ids, tri_n, out_html)


def write_plotly_full(
    pos: np.ndarray,
    titles: list[str],
    paper_ids: list[str],
    tri_n: np.ndarray,
    out_html: Path,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed; skipping interactive HTML", file=sys.stderr)
        return

    hover = [
        f"<b>{truncate(t, 120)}</b><br>paper_id: {pid}<br>triangles: {int(tc)}"
        for t, pid, tc in zip(titles, paper_ids, tri_n)
    ]
    fig = go.Figure(
        data=[
            go.Scattergl(
                x=pos[:, 0],
                y=pos[:, 1],
                mode="markers",
                marker=dict(
                    size=3 + 4 * (np.log1p(tri_n) / (np.log1p(tri_n).max() + 1e-8)),
                    color=tri_n,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="triangle count"),
                ),
                text=hover,
                hoverinfo="text",
            )
        ]
    )
    fig.update_layout(
        title="Full chunk — paper titles on hover (simplicial complex nodes)",
        template="plotly_dark",
        width=1400,
        height=900,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"Saved {out_html}")


def plot_local_showcase(
    record: dict,
    nodes: set[int],
    stacked: np.ndarray | None,
    out_path: Path,
    *,
    title: str,
) -> None:
    edges = [tuple(sorted(e)) for e in record["edges"]]
    edge_w = {tuple(sorted(e)): w for e, w in zip(record["edges"], record["edge_weights"])}
    triangles = [tuple(sorted(t)) for t in record["triangles"]]
    titles = list(record["title"])

    sub_edges = [(u, v) for u, v in edges if u in nodes and v in nodes]
    sub_tris = [t for t in triangles if all(x in nodes for x in t)]
    pos = layout_positions_local(nodes, sub_edges, stacked)

    fig, ax = plt.subplots(figsize=(16, 12), facecolor="#fafafa")
    ax.set_facecolor("#fafafa")

    if sub_tris:
        polys = [[pos[a], pos[b], pos[c]] for a, b, c in sub_tris]
        tri_colors = []
        for a, b, c in sub_tris:
            ws = [
                edge_w.get(tuple(sorted((a, b))), 0.0),
                edge_w.get(tuple(sorted((b, c))), 0.0),
                edge_w.get(tuple(sorted((a, c))), 0.0),
            ]
            tri_colors.append(max(ws))
        ax.add_collection(
            PolyCollection(
                polys,
                facecolors=plt.cm.YlOrRd(np.clip(tri_colors, 0, 1) * 0.35 + 0.08),
                edgecolors="#c44e52",
                linewidths=0.6,
                alpha=0.55,
                zorder=1,
            )
        )

    if sub_edges:
        segs = [[pos[u], pos[v]] for u, v in sub_edges]
        weights = [edge_w.get((u, v), 0.0) for u, v in sub_edges]
        ax.add_collection(
            LineCollection(
                segs,
                colors=plt.cm.Blues(0.35 + 0.45 * np.clip(weights, 0, 1)),
                linewidths=0.8 + 1.2 * np.clip(weights, 0, 1),
                alpha=0.85,
                zorder=2,
            )
        )

    for n in sorted(nodes):
        p = pos[n]
        ax.scatter(p[0], p[1], s=120, c="#2c3e50", edgecolors="white", linewidths=1.2, zorder=3)
        ax.text(
            p[0],
            p[1] + 0.02,
            truncate(titles[n]),
            fontsize=7,
            ha="center",
            va="bottom",
            color="#1a1a1a",
            zorder=4,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#cccccc", alpha=0.92),
        )

    ax.set_title(title, fontsize=13, pad=12)
    ax.axis("off")
    ax.autoscale()
    ax.margins(0.12)
    fig.text(
        0.5,
        0.02,
        f"Local patch: {len(nodes)} nodes, {len(sub_edges)} edges, {len(sub_tris)} triangles",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=EXTRACTED_DIR / "run_10k")
    p.add_argument("--method", default="clique")
    p.add_argument(
        "--scope",
        choices=("full", "local"),
        default="full",
        help="full = entire deduped chunk; local = small labeled subgraph",
    )
    p.add_argument("--max-nodes", type=int, default=36, help="For scope=local only")
    p.add_argument("--seed-node", type=int, default=None)
    p.add_argument("--label-top", type=int, default=40, help="Title labels on highest-triangle nodes (full scope)")
    p.add_argument("--edge-alpha", type=float, default=0.04, help="Edge opacity for full-scope plot")
    p.add_argument("--no-html", action="store_true", help="Skip Plotly hover HTML")
    args = p.parse_args()

    rec_path = args.run_dir / "complexes" / args.method / "complex.pkl"
    if not rec_path.is_file():
        print(f"Missing {rec_path}", file=sys.stderr)
        return 1

    record = load_record(rec_path)
    stacked = load_deduped_stacked(args.run_dir)
    method_label = args.method.replace("_", " ").title()
    fig_dir = args.run_dir / "figures"

    if args.scope == "full":
        if stacked is None:
            print("Need subsample/stacked.npy and dedup/keep_mask.npy for full layout", file=sys.stderr)
            return 1
        out_png = fig_dir / f"{args.method}_full_chunk.png"
        out_html = None if args.no_html else fig_dir / f"{args.method}_full_chunk.html"
        plot_full_chunk(
            record,
            stacked,
            out_png,
            out_html,
            label_top=args.label_top,
            edge_alpha=args.edge_alpha,
            title=f"Simplicial complex — whole chunk ({method_label})",
        )
        meta = {
            "scope": "full",
            "method": args.method,
            "n_nodes": record["n_nodes"],
            "n_edges": len(record["edges"]),
            "n_triangles": len(record["triangles"]),
            "png": str(out_png),
            "html": str(out_html) if out_html else None,
        }
        meta_path = fig_dir / f"{args.method}_full_viz_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return 0

    edges = {tuple(sorted(e)) for e in record["edges"]}
    triangles = [tuple(sorted(t)) for t in record["triangles"]]
    nodes = select_triangle_dense_nodes(
        triangles, edges, max_nodes=args.max_nodes, seed=args.seed_node
    )
    out = fig_dir / f"{args.method}_showcase.png"
    plot_local_showcase(
        record,
        nodes,
        stacked,
        out,
        title=f"Simplicial showcase ({method_label}) — local patch",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
