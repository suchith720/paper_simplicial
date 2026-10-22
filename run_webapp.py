#!/usr/bin/env python3
"""
Paper Simplicial Discovery — interactive web application.

Features:
  - Top 25 Gemini evaluations and synthesized hypotheses
  - Top 1,000 triad explorer (API-backed full dataset)
  - Paper search + kNN neighborhood graph with filled triangle highlights
  - Architecture / pipeline article view
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import (  # noqa: E402
    EVALUATIONS_DIR,
    EXECUTIVE_DIGEST_JSON,
    EXTERNAL_RUN_DIR,
    RESULTS_DIR,
    TOP_1000_CSV,
)

WEBAPP_DIR = ROOT / "webapp"
REVIEW_DIR = RESULTS_DIR / "review_top25"
PIPELINE_ARTICLE_PATH = WEBAPP_DIR / "pipeline_article.json"

PIPELINE_ARTICLE: dict = {}
if PIPELINE_ARTICLE_PATH.is_file():
    with PIPELINE_ARTICLE_PATH.open("r", encoding="utf-8") as f:
        PIPELINE_ARTICLE = json.load(f)

print("Loading simplicial discovery datasets…", flush=True)

# --- Top 1000 triads ---
TRIADS_1000: list[dict] = []
if TOP_1000_CSV.is_file():
    with TOP_1000_CSV.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            TRIADS_1000.append(
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
print(f"  {len(TRIADS_1000)} ranked triads", flush=True)

# --- Evaluations (optional tab) ---
EVAL_PATH = EVALUATIONS_DIR / "top25_evaluations_master.json"
EVALUATIONS: list[dict] = []
if EVAL_PATH.is_file():
    with EVAL_PATH.open("r", encoding="utf-8") as f:
        EVALUATIONS = json.load(f).get("evaluations", [])
DIGESTS_BY_RANK: dict[int, dict] = {}
if EXECUTIVE_DIGEST_JSON.is_file():
    with EXECUTIVE_DIGEST_JSON.open("r", encoding="utf-8") as f:
        for t in json.load(f).get("triads", []):
            DIGESTS_BY_RANK[int(t["rank"])] = t
COMBINED_EVALS = [{**ev, "digest": DIGESTS_BY_RANK.get(ev.get("triad_id"), {})} for ev in EVALUATIONS]

# --- kNN graph + paper index ---
TITLES: list[str] = []
PAPER_IDS: list[str] = []
ROW_IDX: np.ndarray | None = None
KNn_IDX: np.ndarray | None = None
KNn_SIM: np.ndarray | None = None
ADJ: dict[int, set[int]] = {}
EDGE_WEIGHT: dict[tuple[int, int], float] = {}
N_NODES = 0

GRAPH_PKL = EXTERNAL_RUN_DIR / "graph" / "knn_graph.pkl"
KNn_IDX_PATH = EXTERNAL_RUN_DIR / "graph" / "knn_idx.npy"
KNn_SIM_PATH = EXTERNAL_RUN_DIR / "graph" / "knn_sim.npy"
META_PATH = EXTERNAL_RUN_DIR / "meta.json"

RUN_META: dict = {}
if META_PATH.is_file():
    with META_PATH.open("r", encoding="utf-8") as f:
        RUN_META = json.load(f)

if GRAPH_PKL.is_file():
    with GRAPH_PKL.open("rb") as fp:
        gdata = pickle.load(fp)
    titles_raw = gdata["title"]
    paper_ids_raw = gdata["paper_id"]
    TITLES = [str(t) for t in titles_raw]
    PAPER_IDS = [str(p) for p in paper_ids_raw]
    ROW_IDX = np.asarray(gdata["row_idx"], dtype=np.int64)
    N_NODES = len(TITLES)
    for u, v, w in gdata["edges"]:
        u, v = int(u), int(v)
        key = (u, v) if u < v else (v, u)
        EDGE_WEIGHT[key] = float(w)
        ADJ.setdefault(u, set()).add(v)
        ADJ.setdefault(v, set()).add(u)
    print(f"  Graph: {N_NODES} nodes, {len(EDGE_WEIGHT)} edges", flush=True)
else:
    print(f"  WARNING: graph not found at {GRAPH_PKL}", flush=True)

if KNn_IDX_PATH.is_file() and KNn_SIM_PATH.is_file():
    KNn_IDX = np.load(KNn_IDX_PATH)
    KNn_SIM = np.load(KNn_SIM_PATH)
    print(f"  kNN arrays: {KNn_IDX.shape}", flush=True)

_WORD = re.compile(r"[a-z0-9]+")


def _score_paper_query(query: str, node: int) -> float:
    q = query.strip().lower()
    if not q or node < 0 or node >= N_NODES:
        return 0.0
    title = TITLES[node].lower()
    pid = PAPER_IDS[node].lower()
    if q.isdigit():
        if q == str(node):
            return 10_000.0
        if q == pid:
            return 9_000.0
        if pid.startswith(q) or q in pid:
            return 5_000.0
    if q in title:
        return 3_000.0 + len(q)
    q_tokens = _WORD.findall(q)
    if not q_tokens:
        return 0.0
    title_tokens = set(_WORD.findall(title))
    overlap = sum(1 for t in q_tokens if t in title_tokens)
    if overlap == 0:
        return 0.0
    return overlap * 100.0 + overlap / len(q_tokens) * 50.0


def search_papers(query: str, limit: int = 40) -> list[dict]:
    limit = max(1, min(100, limit))
    if not query.strip() or N_NODES == 0:
        return []
    scored: list[tuple[float, int]] = []
    for i in range(N_NODES):
        s = _score_paper_query(query, i)
        if s > 0:
            scored.append((s, i))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out = []
    for s, i in scored[:limit]:
        out.append(
            {
                "node": i,
                "paper_id": PAPER_IDS[i],
                "row_idx": int(ROW_IDX[i]) if ROW_IDX is not None else None,
                "title": TITLES[i],
                "match_score": round(s, 2),
            }
        )
    return out


def expand_neighborhood(center: int, n_neighbors: int) -> set[int]:
    """Center + up to n_neighbors other nodes (n_neighbors capped at 20)."""
    n_neighbors = max(1, min(20, int(n_neighbors)))
    if center < 0 or center >= N_NODES:
        return set()
    selected = {center}
    if KNn_IDX is None or KNn_SIM is None:
        return selected

    direct: list[tuple[float, int]] = []
    for j in range(KNn_IDX.shape[1]):
        nb = int(KNn_IDX[center, j])
        if nb != center:
            direct.append((float(KNn_SIM[center, j]), nb))
    direct.sort(key=lambda x: -x[0])

    for _, nb in direct:
        if len(selected) >= 1 + n_neighbors:
            break
        selected.add(nb)

    if len(selected) >= 1 + n_neighbors:
        return selected

    second_hop: list[tuple[float, int]] = []
    for sim1, nb1 in direct:
        for j in range(KNn_IDX.shape[1]):
            nb2 = int(KNn_IDX[nb1, j])
            if nb2 == center:
                continue
            sim2 = float(KNn_SIM[nb1, j])
            second_hop.append((sim1 * sim2, nb2))
    second_hop.sort(key=lambda x: -x[0])
    for _, nb in second_hop:
        if len(selected) >= 1 + n_neighbors:
            break
        selected.add(nb)
    return selected


def induced_subgraph(nodes: set[int]) -> tuple[list[dict], list[dict], list[list[int]]]:
    node_list = sorted(nodes)
    id_map = {n: i for i, n in enumerate(node_list)}
    edges_out: list[dict] = []
    for u in node_list:
        for v in ADJ.get(u, ()):
            if v in nodes and u < v:
                key = (u, v)
                w = EDGE_WEIGHT.get(key, EDGE_WEIGHT.get((v, u), 0.0))
                edges_out.append({"from": u, "to": v, "weight": round(w, 4)})

    triangles: list[list[int]] = []
    for i, a in enumerate(node_list):
        for j in range(i + 1, len(node_list)):
            b = node_list[j]
            if b not in ADJ.get(a, ()):
                continue
            for k in range(j + 1, len(node_list)):
                c = node_list[k]
                if c in ADJ.get(a, ()) and c in ADJ.get(b, ()):
                    triangles.append([a, b, c])

    nodes_out = [
        {
            "id": n,
            "paper_id": PAPER_IDS[n],
            "row_idx": int(ROW_IDX[n]) if ROW_IDX is not None else None,
            "title": TITLES[n],
            "is_center": False,
        }
        for n in node_list
    ]
    return nodes_out, edges_out, triangles


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEBAPP_DIR), **kwargs)

    def log_message(self, fmt, *args):
        pass

    def send_json(self, data: object, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/api/stats":
            synergy_count = sum(
                1 for e in EVALUATIONS if "Genuine" in e.get("evaluation", {}).get("verdict", "")
            )
            overlap_count = sum(
                1 for e in EVALUATIONS if "Moderate" in e.get("evaluation", {}).get("verdict", "")
            )
            spurious_count = sum(
                1 for e in EVALUATIONS if "Spurious" in e.get("evaluation", {}).get("verdict", "")
            )
            scores = [
                e.get("evaluation", {}).get("triad_quality_score", 0)
                for e in EVALUATIONS
                if "triad_quality_score" in e.get("evaluation", {})
            ]
            avg_score = round(sum(scores) / max(len(scores), 1), 2)
            return self.send_json(
                {
                    "n_papers": RUN_META.get("n_papers_out", N_NODES),
                    "n_edges": RUN_META.get("n_edges", len(EDGE_WEIGHT)),
                    "n_simplices_total": 94180,
                    "n_ranked_triads": len(TRIADS_1000),
                    "n_evaluations": len(EVALUATIONS),
                    "n_hypotheses": len(EVALUATIONS) * 5,
                    "avg_synergy_score": avg_score,
                    "verdict_breakdown": {
                        "genuine_synergy": synergy_count,
                        "moderate_overlap": overlap_count,
                        "spurious": spurious_count,
                    },
                    "knn_k": RUN_META.get("k", 5),
                }
            )

        if path == "/api/triads/all":
            return self.send_json({"triads": TRIADS_1000, "total": len(TRIADS_1000)})

        if path == "/api/triads":
            query = params.get("q", [""])[0].lower()
            min_score = float(params.get("min_score", [0.0])[0])
            page = max(1, int(params.get("page", [1])[0]))
            limit = min(100, max(5, int(params.get("limit", [20])[0])))
            filtered = TRIADS_1000
            if min_score > 0:
                filtered = [t for t in filtered if t["score"] >= min_score]
            if query:
                filtered = [
                    t
                    for t in filtered
                    if query in t["title_u"].lower()
                    or query in t["title_v"].lower()
                    or query in t["title_w"].lower()
                    or query in t["paper_id_u"]
                    or query in t["paper_id_v"]
                    or query in t["paper_id_w"]
                ]
            total = len(filtered)
            start = (page - 1) * limit
            page_items = filtered[start : start + limit]
            return self.send_json(
                {
                    "page": page,
                    "limit": limit,
                    "total": total,
                    "total_pages": max(1, (total + limit - 1) // limit),
                    "triads": page_items,
                }
            )

        if path == "/api/papers/search":
            q = params.get("q", [""])[0]
            limit = int(params.get("limit", [40])[0])
            return self.send_json({"query": q, "results": search_papers(q, limit)})

        if path == "/api/graph/explore":
            try:
                center = int(params.get("node", ["-1"])[0])
                n_neighbors = int(params.get("neighbors", ["5"])[0])
            except ValueError:
                return self.send_json({"error": "invalid node or neighbors"}, HTTPStatus.BAD_REQUEST)
            if center < 0 or center >= N_NODES:
                return self.send_json({"error": "node out of range"}, HTTPStatus.NOT_FOUND)
            nodes_set = expand_neighborhood(center, n_neighbors)
            nodes_out, edges_out, tris = induced_subgraph(nodes_set)
            for n in nodes_out:
                n["is_center"] = n["id"] == center
            return self.send_json(
                {
                    "center_node": center,
                    "n_neighbors_requested": min(20, max(1, n_neighbors)),
                    "n_nodes": len(nodes_out),
                    "n_edges": len(edges_out),
                    "n_triangles": len(tris),
                    "nodes": nodes_out,
                    "edges": edges_out,
                    "triangles": tris,
                }
            )

        if path == "/api/evaluations":
            verdict_filter = params.get("verdict", [None])[0]
            search_query = params.get("q", [""])[0].lower()
            results = COMBINED_EVALS
            if verdict_filter and verdict_filter != "all":
                results = [
                    e
                    for e in results
                    if verdict_filter.lower() in e.get("evaluation", {}).get("verdict", "").lower()
                ]
            if search_query:
                results = [
                    e for e in results if search_query in json.dumps(e, ensure_ascii=False).lower()
                ]
            return self.send_json({"evaluations": results, "total": len(results)})

        if path == "/data/evaluations.json":
            fallback = ROOT / "webapp" / "data" / "evaluations.json"
            if fallback.is_file():
                body = fallback.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            return self.send_json({"evaluations": COMBINED_EVALS})

        if path.startswith("/api/triad/"):
            try:
                tid = int(path.split("/")[-1])
                triad = next((t for t in TRIADS_1000 if t["rank"] == tid), None)
                if not triad:
                    return self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                ev = next((e for e in EVALUATIONS if e.get("triad_id") == tid), None)
                return self.send_json(
                    {"triad": triad, "digest": DIGESTS_BY_RANK.get(tid), "evaluation": ev}
                )
            except ValueError:
                return self.send_json({"error": "bad id"}, HTTPStatus.BAD_REQUEST)

        if path == "/api/pipeline":
            synergy_count = sum(
                1 for e in EVALUATIONS if "Genuine" in e.get("evaluation", {}).get("verdict", "")
            )
            moderate_count = sum(
                1 for e in EVALUATIONS if "Moderate" in e.get("evaluation", {}).get("verdict", "")
            )
            spurious_count = sum(
                1 for e in EVALUATIONS if "Spurious" in e.get("evaluation", {}).get("verdict", "")
            )
            payload = dict(PIPELINE_ARTICLE)
            payload["metrics"] = {
                "input_papers": RUN_META.get("n_papers_in", 45666),
                "dedup_papers": RUN_META.get("n_papers_out", N_NODES),
                "edges": RUN_META.get("n_edges", len(EDGE_WEIGHT)),
                "triangles": 94180,
                "ranked_triads": len(TRIADS_1000),
                "reviewed_triads": len(EVALUATIONS),
                "hypotheses": len(EVALUATIONS) * 5,
                "verdict_genuine": synergy_count,
                "verdict_moderate": moderate_count,
                "verdict_spurious": spurious_count,
            }
            return self.send_json(payload)

        if path in ("", "/"):
            self.path = "/index.html"
        return super().do_GET()


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper Simplicial Discovery web app")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    if not WEBAPP_DIR.is_dir():
        print(f"Missing {WEBAPP_DIR}", file=sys.stderr)
        return 1

    httpd = ThreadingHTTPServer((args.host, args.port), AppHandler)
    host = "localhost" if args.host == "0.0.0.0" else args.host
    print(f"\nPaper Simplicial Discovery → http://{host}:{args.port}\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
