#!/usr/bin/env python3
"""
KMeans clusters on English-science embeddings; simplicial extraction per cluster.

Reads extracted_simplicial/english_science/manifest.npz, assigns each paper a
cluster (MiniBatchKMeans on L2-normalized windows_mean), then for each cluster
with at least --min-papers runs dedupe -> kNN -> clique/neighbor lifts.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.dedupe import merge_clusters_at_threshold  # noqa: E402
from lib.embeddings import load_manifest_batch  # noqa: E402
from lib.graph import knn_undirected_graph  # noqa: E402
from lib.io import PaperSimplicialRecord, save_graph_pickle, save_simplicial_record, write_json  # noqa: E402
from lib.simplicial import lift_graph  # noqa: E402
from project_paths import EMBEDDINGS_DIR, ENGLISH_SCIENCE_MANIFEST  # noqa: E402

DEFAULT_METHODS = ("clique", "neighbor")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def run_one_cluster(
    *,
    cluster_id: int,
    batch,
    member_idx: np.ndarray,
    manifest: np.lib.npyio.NpzFile,
    out_root: Path,
    dedupe_threshold: float,
    k: int,
    methods: tuple[str, ...],
    dedupe_block: int,
) -> dict | None:
    n = len(member_idx)
    if n < 2:
        return None

    x = batch.stacked[member_idx]
    row_idx = batch.row_idx[member_idx]
    paper_id = batch.paper_id[member_idx]
    title = batch.title[member_idx]
    global_idx = np.asarray(manifest["global_idx"], dtype=np.int64)[member_idx]

    _cluster_id, keep, dedup_stats = merge_clusters_at_threshold(
        x, dedupe_threshold, block=dedupe_block
    )
    row_idx = row_idx[keep]
    paper_id = paper_id[keep]
    title = title[keep]
    global_idx = global_idx[keep]
    x = x[keep]
    n_out = int(keep.sum())
    if n_out < 2:
        return {"cluster_id": cluster_id, "skipped": "too_few_after_dedupe", "n_in": n}

    k_eff = min(k, n_out - 1)
    if k_eff < 1:
        return {"cluster_id": cluster_id, "skipped": "knn_impossible", "n_in": n}

    G, knn_idx, knn_sim, mean = knn_undirected_graph(x, k_eff)
    run_dir = out_root / f"cluster_{cluster_id:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    graph_edges = [(int(u), int(v), float(d.get("weight", 0.0))) for u, v, d in G.edges(data=True)]
    graph_meta = {
        "embedding_cluster": cluster_id,
        "k": k_eff,
        "symmetrization": "union",
        "centering": True,
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
    }
    save_graph_pickle(
        run_dir / "graph" / "knn_graph.pkl",
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

    graph_weights = {tuple(sorted((u, v))): float(d.get("weight", 0.0)) for u, v, d in G.edges(data=True)}
    lift_summary: dict[str, dict[str, int]] = {}
    for method in methods:
        edges, triangles, vr_r = lift_graph(G, method, max_rank=2, max_dim=2)  # type: ignore[arg-type]
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
        out_pkl = run_dir / "complexes" / method / "complex.pkl"
        save_simplicial_record(record, out_pkl)
        canon_tris = [tuple(sorted(t)) for t in triangles]
        np.savez_compressed(
            run_dir / "complexes" / method / "skeleton.npz",
            edges=np.asarray(canon_edges, dtype=np.int32),
            edge_weights=np.asarray(edge_weights, dtype=np.float32),
            triangles=np.asarray(canon_tris, dtype=np.int32).reshape(-1, 3)
            if canon_tris
            else np.empty((0, 3), dtype=np.int32),
        )
        lift_summary[method] = {"n_edges": len(canon_edges), "n_triangles": len(triangles)}

    meta = {
        "cluster_id": cluster_id,
        "n_manifest": n,
        **dedup_stats,
        **graph_meta,
        "lifts": lift_summary,
    }
    write_json(run_dir / "meta.json", meta)
    return meta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=ENGLISH_SCIENCE_MANIFEST)
    p.add_argument("--embeddings-dir", type=Path, default=EMBEDDINGS_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n-clusters", type=int, default=14)
    p.add_argument("--min-papers", type=int, default=300, help="Skip clusters smaller than this")
    p.add_argument("--dedupe-threshold", type=float, default=0.99)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    p.add_argument("--dedupe-block", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_root = args.out_dir or args.manifest.parent.parent / "run_english_science_clusterwise"
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    t0 = time.time()

    manifest = np.load(args.manifest, allow_pickle=True)
    print(f"Loading manifest ({len(manifest['global_idx']):,} papers) …", flush=True)
    batch = load_manifest_batch(args.manifest, embeddings_dir=args.embeddings_dir)
    n = len(batch.row_idx)

    feat = l2_normalize(batch.windows_mean.astype(np.float64))
    k = min(int(args.n_clusters), max(n, 1))
    print(f"MiniBatchKMeans k={k} on windows_mean …", flush=True)
    km = MiniBatchKMeans(
        n_clusters=k,
        batch_size=8192,
        random_state=args.seed,
        n_init=5,
        max_iter=100,
    )
    labels = km.fit_predict(feat).astype(np.int16)

    label_path = args.manifest.parent / "kmeans_labels.npz"
    np.savez_compressed(
        label_path,
        cluster_id=labels,
        global_idx=np.asarray(manifest["global_idx"], dtype=np.int64),
        row_idx=batch.row_idx,
        n_clusters=np.array([k], dtype=np.int16),
        seed=np.array([args.seed], dtype=np.int32),
    )
    print(f"Saved cluster labels -> {label_path}", flush=True)

    summaries = []
    for cid in range(k):
        member = np.flatnonzero(labels == cid)
        print(f"\nCluster {cid:02d}: {len(member):,} papers", flush=True)
        if len(member) < args.min_papers:
            print(f"  skip (< {args.min_papers})", flush=True)
            summaries.append({"cluster_id": cid, "n_manifest": int(len(member)), "skipped": "below_min_papers"})
            continue
        meta = run_one_cluster(
            cluster_id=cid,
            batch=batch,
            member_idx=member,
            manifest=manifest,
            out_root=out_root,
            dedupe_threshold=args.dedupe_threshold,
            k=args.k,
            methods=methods,
            dedupe_block=args.dedupe_block,
        )
        if meta:
            summaries.append(meta)
            if "n_edges" in meta:
                print(f"  |V|={meta['n_nodes']}, |E|={meta['n_edges']}", flush=True)

    write_json(
        out_root / "meta.json",
        {
            "manifest": str(args.manifest),
            "n_clusters": k,
            "min_papers": args.min_papers,
            "label_npz": str(label_path),
            "clusters": summaries,
            "elapsed_sec": round(time.time() - t0, 1),
        },
    )
    print(f"\nDone in {time.time() - t0:.1f}s. Root: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
