#!/usr/bin/env python3
"""
Simplicial extraction for a manifest of papers (e.g. English science subset).

Loads embeddings by chunk_id + chunk_local_idx, then dedupe -> kNN -> complexes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.dedupe import merge_clusters_at_threshold  # noqa: E402
from lib.embeddings import load_manifest_batch  # noqa: E402
from lib.graph import knn_undirected_graph  # noqa: E402
from lib.io import PaperSimplicialRecord, save_graph_pickle, save_simplicial_record, write_json  # noqa: E402
from lib.simplicial import lift_graph  # noqa: E402
from project_paths import EMBEDDINGS_DIR, ENGLISH_SCIENCE_MANIFEST, ENGLISH_SCIENCE_RUN  # noqa: E402

DEFAULT_METHODS = ("clique", "neighbor")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=ENGLISH_SCIENCE_MANIFEST)
    p.add_argument("--embeddings-dir", type=Path, default=EMBEDDINGS_DIR)
    p.add_argument("--out-dir", type=Path, default=ENGLISH_SCIENCE_RUN)
    p.add_argument("--dedupe-threshold", type=float, default=0.99)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    p.add_argument("--max-rank", type=int, default=2)
    p.add_argument("--max-dim", type=int, default=2)
    p.add_argument("--dedupe-block", type=int, default=2000)
    args = p.parse_args()

    if not args.manifest.is_file():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        print("Run scripts/build_english_science_manifest.py first.", file=sys.stderr)
        return 1

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    run_dir = args.out_dir
    t0 = time.time()

    m = np.load(args.manifest, allow_pickle=True)
    print(f"Loading manifest ({len(m['global_idx']):,} papers) …", flush=True)
    batch = load_manifest_batch(args.manifest, embeddings_dir=args.embeddings_dir)
    n_in = len(batch.row_idx)
    print(f"  papers: {n_in}, stacked dim: {batch.stacked.shape[1]}", flush=True)

    subsample_dir = run_dir / "subsample"
    subsample_dir.mkdir(parents=True, exist_ok=True)
    np.save(subsample_dir / "stacked.npy", batch.stacked)
    np.save(subsample_dir / "row_idx.npy", batch.row_idx)
    np.save(subsample_dir / "global_idx.npy", m["global_idx"])
    np.save(subsample_dir / "chunk_id.npy", m["chunk_id"])
    np.save(subsample_dir / "chunk_local_idx.npy", m["chunk_local_idx"])
    write_json(
        subsample_dir / "meta.json",
        {
            "manifest": str(args.manifest),
            "n_papers": n_in,
            "feature": "concat(head, windows_mean)",
        },
    )

    print(f"Dedupe at cosine >= {args.dedupe_threshold} …", flush=True)
    cluster_id, keep, dedup_stats = merge_clusters_at_threshold(
        batch.stacked, args.dedupe_threshold, block=args.dedupe_block
    )
    dedup_dir = run_dir / "dedup"
    dedup_dir.mkdir(parents=True, exist_ok=True)
    np.save(dedup_dir / "cluster_id.npy", cluster_id)
    np.save(dedup_dir / "keep_mask.npy", keep)
    write_json(dedup_dir / "dedup_report.json", dedup_stats)
    print(f"  {dedup_stats['n_papers_in']} -> {dedup_stats['n_papers_out']} papers", flush=True)

    row_idx = batch.row_idx[keep]
    paper_id = batch.paper_id[keep]
    title = batch.title[keep]
    global_idx = np.asarray(m["global_idx"], dtype=np.int64)[keep]
    x = batch.stacked[keep]

    print(f"kNN k={args.k} (centered cosine, union symmetrization) …", flush=True)
    G, knn_idx, knn_sim, mean = knn_undirected_graph(x, args.k)
    graph_edges = [(int(u), int(v), float(d.get("weight", 0.0))) for u, v, d in G.edges(data=True)]
    graph_meta = {
        "k": args.k,
        "symmetrization": "union",
        "centering": True,
        "remove_top_pcs": 0,
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
    }
    graph_path = run_dir / "graph" / "knn_graph.pkl"
    save_graph_pickle(
        graph_path,
        edges=graph_edges,
        row_idx=row_idx,
        paper_id=paper_id,
        title=title,
        centering_mean=mean,
        meta=graph_meta,
    )
    np.save(run_dir / "graph" / "knn_idx.npy", knn_idx)
    np.save(run_dir / "graph" / "knn_sim.npy", knn_sim)
    np.save(run_dir / "graph" / "global_idx.npy", global_idx)
    print(f"  graph |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}", flush=True)

    graph_weights = {tuple(sorted((u, v))): float(d.get("weight", 0.0)) for u, v, d in G.edges(data=True)}
    complexes_dir = run_dir / "complexes"
    run_meta = {
        "manifest": str(args.manifest),
        "dedupe_threshold": args.dedupe_threshold,
        **dedup_stats,
        **graph_meta,
        "methods": list(methods),
        "max_rank": args.max_rank,
        "max_dim": args.max_dim,
        "elapsed_sec": None,
    }

    for method in methods:
        print(f"Lift: {method} …", flush=True)
        edges, triangles, vr_r = lift_graph(
            G,
            method,  # type: ignore[arg-type]
            max_rank=args.max_rank,
            max_dim=args.max_dim,
        )
        canon_edges = [tuple(sorted(e)) for e in edges]
        edge_weights = tuple(graph_weights.get(e, 0.0) for e in canon_edges)
        record = PaperSimplicialRecord(
            method=method,
            n_nodes=G.number_of_nodes(),
            n_edges_graph=G.number_of_edges(),
            edges=tuple(canon_edges),
            edge_weights=edge_weights,
            triangles=tuple(tuple(sorted(t)) for t in triangles),
            row_idx=tuple(int(v) for v in row_idx),
            paper_id=tuple(str(v) for v in paper_id),
            title=tuple(str(v) for v in title),
            vr_radius=vr_r,
        )
        out_pkl = complexes_dir / method / "complex.pkl"
        save_simplicial_record(record, out_pkl)
        canon_tris = [tuple(sorted(t)) for t in triangles]
        np.savez_compressed(
            complexes_dir / method / "skeleton.npz",
            edges=np.asarray(canon_edges, dtype=np.int32),
            edge_weights=np.asarray(edge_weights, dtype=np.float32),
            triangles=np.asarray(canon_tris, dtype=np.int32).reshape(-1, 3)
            if canon_tris
            else np.empty((0, 3), dtype=np.int32),
        )
        print(
            f"  saved {len(canon_edges)} edges, {len(triangles)} triangles -> "
            f"{out_pkl.name} + skeleton.npz",
            flush=True,
        )

    run_meta["elapsed_sec"] = round(time.time() - t0, 1)
    write_json(run_dir / "meta.json", run_meta)
    print(f"\nDone in {run_meta['elapsed_sec']}s. Outputs: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
