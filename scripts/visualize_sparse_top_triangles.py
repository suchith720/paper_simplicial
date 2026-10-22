#!/usr/bin/env python3
"""
All deduped papers (~9.7k): small nodes + only top-ranked triangles (sparse, readable).

Triangles ranked by semantic tightness (default: mean×min edge cosine).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import EXTRACTED_DIR  # noqa: E402
from lib.triangle_rank import rank_triangles  # noqa: E402

NODE_COLOR = "#cbd5e1"
NODE_SIZE = 6
TRI_CMAP = plt.cm.magma


def layout_pca2(stacked: np.ndarray) -> np.ndarray:
    X = stacked.astype(np.float64)
    X -= X.mean(axis=0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return X @ vt[:2].T


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=EXTRACTED_DIR / "run_10k")
    p.add_argument("--method", choices=("clique", "neighbor"), default="clique")
    p.add_argument("--top-triangles", type=int, default=400)
    p.add_argument(
        "--score",
        choices=("min", "mean", "mean_min", "geom", "geom_min", "english_tight"),
        default="english_tight",
    )
    p.add_argument("--min-corpus-en", type=float, default=0.0)
    p.add_argument("--min-title-en", type=float, default=0.0)
    p.add_argument("--min-title-latin", type=float, default=0.0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    rec = pickle.load(open(args.run_dir / "complexes" / args.method / "complex.pkl", "rb"))["record"]
    full = np.load(args.run_dir / "subsample/stacked.npy")
    keep = np.load(args.run_dir / "dedup/keep_mask.npy").astype(bool)
    stacked = full[keep]
    pos = layout_pca2(stacked)
    n = pos.shape[0]

    ranked = rank_triangles(
        rec,
        score_mode=args.score,
        min_corpus_en=args.min_corpus_en,
        min_title_en=args.min_title_en,
        min_title_latin=args.min_title_latin,
        english_only=(args.score == "english_tight"),
    )
    top = ranked[: min(args.top_triangles, len(ranked))]
    if not top:
        print("No triangles passed filters / ranking.", file=sys.stderr)
        return 1

    scores = np.array([t.score for t in top], dtype=np.float64)
    smin, smax = scores.min(), scores.max()
    norm = (scores - smin) / (smax - smin + 1e-8)

    fig, ax = plt.subplots(figsize=(20, 16), facecolor="white")
    ax.set_facecolor("#f8fafc")

    polys = []
    facecolors = []
    for t, nv in zip(top, norm):
        u, v, w = t.nodes
        polys.append(np.array([pos[u], pos[v], pos[w]]))
        facecolors.append(TRI_CMAP(0.15 + 0.75 * nv))

    ax.add_collection(
        PolyCollection(
            polys,
            facecolors=facecolors,
            edgecolors="#7f1d1d",
            linewidths=0.6,
            alpha=0.72,
            zorder=2,
        )
    )

    # Highlight nodes that appear in at least one top triangle
    in_top = set()
    for t in top:
        in_top.update(t.nodes)
    in_top_arr = np.array(sorted(in_top), dtype=np.int64)
    rest = np.setdiff1d(np.arange(n), in_top_arr)

    if rest.size:
        ax.scatter(
            pos[rest, 0],
            pos[rest, 1],
            s=NODE_SIZE,
            c=NODE_COLOR,
            alpha=0.35,
            edgecolors="none",
            zorder=1,
        )
    ax.scatter(
        pos[in_top_arr, 0],
        pos[in_top_arr, 1],
        s=NODE_SIZE * 2.2,
        c="#0f172a",
        alpha=0.85,
        edgecolors="white",
        linewidths=0.25,
        zorder=3,
    )

    subtitle = f"score={args.score}"
    if args.score == "english_tight":
        subtitle += " (English titles + corpus en, geom×min edge cosine)"
    ax.set_title(
        f"All {n:,} papers — top {len(top)} triangles only ({args.method}, {subtitle})\n"
        f"No edge lines; color = hypothesis rank (brighter = tighter triple)",
        fontsize=14,
        pad=10,
    )
    ax.axis("off")
    ax.autoscale()
    ax.margins(0.02)

    legend = [
        mpatches.Patch(facecolor=NODE_COLOR, label=f"Other papers ({n - len(in_top):,})"),
        mpatches.Patch(facecolor="#0f172a", label=f"Papers in top-{len(top)} triangles ({len(in_top):,})"),
        mpatches.Patch(facecolor=TRI_CMAP(0.85), edgecolor="#7f1d1d", label="Ranked 2-simplex (3-paper hypothesis)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=10)

    fig.text(
        0.5,
        0.01,
        "Full deduped 10k chunk (~9.7k). Only best triangles drawn for sparsity. See hypotheses/*.csv for titles.",
        ha="center",
        fontsize=9,
        color="#64748b",
    )

    score_tag = args.score if args.score != "english_tight" else "english_tight"
    out = args.out or (
        args.run_dir / "figures" / f"{args.method}_sparse_{score_tag}_top{len(top)}_all_nodes.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")

    meta = {
        "n_nodes": n,
        "top_triangles": len(top),
        "score_mode": args.score,
        "n_triangles_ranked": len(ranked),
        "nodes_in_top_triangles": len(in_top),
        "output": str(out),
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
