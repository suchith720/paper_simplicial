#!/usr/bin/env python3
"""Export JSON bundles under webapp/data/ for GitHub Pages (static hosting)."""
from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import (  # noqa: E402
    EVALUATIONS_DIR,
    EXECUTIVE_DIGEST_JSON,
    EXTERNAL_RUN_DIR,
    TOP_1000_CSV,
)

WEBAPP_DATA = ROOT / "webapp" / "data"
PIPELINE_ARTICLE_PATH = ROOT / "webapp" / "pipeline_article.json"


def _write(name: str, obj: object) -> None:
    path = WEBAPP_DATA / name
    body = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    path.write_text(body, encoding="utf-8")
    print(f"  {name}: {path.stat().st_size / 1_048_576:.2f} MiB")


def main() -> int:
    WEBAPP_DATA.mkdir(parents=True, exist_ok=True)

    triads: list[dict] = []
    if TOP_1000_CSV.is_file():
        with TOP_1000_CSV.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                triads.append(
                    {
                        "rank": int(r["rank"]),
                        "score": float(r["score"]),
                        "w_uv": float(r["w_uv"]),
                        "w_vw": float(r["w_vw"]),
                        "w_uw": float(r["w_uw"]),
                        "title_u": r["title_u"],
                        "title_v": r["title_v"],
                        "title_w": r["title_w"],
                        "paper_id_u": r["paper_id_u"],
                        "paper_id_v": r["paper_id_v"],
                        "paper_id_w": r["paper_id_w"],
                    }
                )

    eval_path = EVALUATIONS_DIR / "top25_evaluations_master.json"
    evaluations: list[dict] = []
    if eval_path.is_file():
        with eval_path.open("r", encoding="utf-8") as f:
            evaluations = json.load(f).get("evaluations", [])

    digests_by_rank: dict[int, dict] = {}
    if EXECUTIVE_DIGEST_JSON.is_file():
        with EXECUTIVE_DIGEST_JSON.open("r", encoding="utf-8") as f:
            for t in json.load(f).get("triads", []):
                digests_by_rank[int(t["rank"])] = t

    combined = [{**ev, "digest": digests_by_rank.get(ev.get("triad_id"), {})} for ev in evaluations]

    synergy = sum(1 for e in evaluations if "Genuine" in e.get("evaluation", {}).get("verdict", ""))
    moderate = sum(1 for e in evaluations if "Moderate" in e.get("evaluation", {}).get("verdict", ""))
    spurious = sum(1 for e in evaluations if "Spurious" in e.get("evaluation", {}).get("verdict", ""))
    scores = [
        e.get("evaluation", {}).get("triad_quality_score", 0)
        for e in evaluations
        if "triad_quality_score" in e.get("evaluation", {})
    ]
    avg_score = round(sum(scores) / max(len(scores), 1), 2)

    meta_path = EXTERNAL_RUN_DIR / "meta.json"
    run_meta: dict = {}
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            run_meta = json.load(f)

    graph_pkl = EXTERNAL_RUN_DIR / "graph" / "knn_graph.pkl"
    knn_idx_path = EXTERNAL_RUN_DIR / "graph" / "knn_idx.npy"
    knn_sim_path = EXTERNAL_RUN_DIR / "graph" / "knn_sim.npy"

    n_nodes = 0
    n_edges = 0
    graph_payload: dict | None = None

    if graph_pkl.is_file():
        with graph_pkl.open("rb") as fp:
            gdata = pickle.load(fp)
        titles = [str(t) for t in gdata["title"]]
        paper_ids = [str(p) for p in gdata["paper_id"]]
        n_nodes = len(titles)
        edges: list[list[float | int]] = []
        for u, v, w in gdata["edges"]:
            u, v = int(u), int(v)
            if u > v:
                u, v = v, u
            edges.append([u, v, round(float(w), 4)])
        n_edges = len(edges)

        knn_idx = np.load(knn_idx_path).tolist() if knn_idx_path.is_file() else []
        knn_sim = np.load(knn_sim_path).tolist() if knn_sim_path.is_file() else []

        graph_payload = {
            "n_nodes": n_nodes,
            "titles": titles,
            "paper_ids": paper_ids,
            "edges": edges,
            "knn_idx": knn_idx,
            "knn_sim": knn_sim,
        }

    print("Writing webapp/data…")
    _write("triads.json", {"triads": triads, "total": len(triads)})
    _write("evaluations.json", {"evaluations": combined, "total": len(combined)})
    _write(
        "stats.json",
        {
            "n_papers": run_meta.get("n_papers_out", n_nodes),
            "n_edges": run_meta.get("n_edges", n_edges),
            "n_simplices_total": 94180,
            "n_ranked_triads": len(triads),
            "n_evaluations": len(evaluations),
            "n_hypotheses": len(evaluations) * 5,
            "avg_synergy_score": avg_score,
            "verdict_breakdown": {
                "genuine_synergy": synergy,
                "moderate_overlap": moderate,
                "spurious": spurious,
            },
            "knn_k": run_meta.get("k", 5),
        },
    )

    pipeline_article: dict = {}
    if PIPELINE_ARTICLE_PATH.is_file():
        with PIPELINE_ARTICLE_PATH.open("r", encoding="utf-8") as f:
            pipeline_article = json.load(f)
    pipeline_article = dict(pipeline_article)
    pipeline_article["metrics"] = {
        "input_papers": run_meta.get("n_papers_in", 45666),
        "dedup_papers": run_meta.get("n_papers_out", n_nodes),
        "edges": run_meta.get("n_edges", n_edges),
        "triangles": 94180,
        "ranked_triads": len(triads),
        "reviewed_triads": len(evaluations),
        "hypotheses": len(evaluations) * 5,
        "verdict_genuine": synergy,
        "verdict_moderate": moderate,
        "verdict_spurious": spurious,
    }
    _write("pipeline.json", pipeline_article)

    if graph_payload:
        _write("graph.json", graph_payload)
    else:
        print("  WARNING: graph.pkl missing — graph.json not written", file=sys.stderr)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
