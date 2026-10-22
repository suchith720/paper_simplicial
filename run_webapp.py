#!/usr/bin/env python3
"""
Zero-dependency web application server for Paper Simplicial Discovery.
Serves interactive explorer for simplicial triads, Gemini Pro evaluations, and 125 hypotheses.
"""
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
WEBAPP_DIR = ROOT / "webapp"
RESULTS_DIR = ROOT / "results"
REVIEW_DIR = RESULTS_DIR / "review_top25"

# Load data into memory on startup
print("Loading simplicial discovery datasets...", flush=True)

CSV_PATH = RESULTS_DIR / "top_1000_triangles.csv"
TRIADS_1000: list[dict] = []
if CSV_PATH.exists():
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            TRIADS_1000.append({
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
            })
print(f"  Loaded {len(TRIADS_1000)} ranked triads from CSV.", flush=True)

EVAL_PATH = REVIEW_DIR / "evaluations" / "top25_evaluations_master.json"
EVALUATIONS: list[dict] = []
if EVAL_PATH.exists():
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        EVALUATIONS = json.load(f).get("evaluations", [])
print(f"  Loaded {len(EVALUATIONS)} Gemini Pro evaluations.", flush=True)

DIGEST_PATH = REVIEW_DIR / "top25_triads_executive_digest.json"
DIGESTS_BY_RANK: dict[int, dict] = {}
if DIGEST_PATH.exists():
    with open(DIGEST_PATH, "r", encoding="utf-8") as f:
        for t in json.load(f).get("triads", []):
            DIGESTS_BY_RANK[t["rank"]] = t
print(f"  Loaded {len(DIGESTS_BY_RANK)} executive digests.", flush=True)

# Combine evaluations with digests
COMBINED_EVALS = []
for ev in EVALUATIONS:
    tid = ev.get("triad_id")
    digest = DIGESTS_BY_RANK.get(tid, {})
    COMBINED_EVALS.append({
        **ev,
        "digest": digest
    })

class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEBAPP_DIR), **kwargs)

    def log_message(self, format, *args):
        # Quieter logging
        pass

    def send_json(self, data: object, status: int = HTTPStatus.OK):
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

        # API Endpoints
        if path == "/api/stats":
            synergy_count = sum(1 for e in EVALUATIONS if "Genuine" in e.get("evaluation", {}).get("verdict", ""))
            overlap_count = sum(1 for e in EVALUATIONS if "Moderate" in e.get("evaluation", {}).get("verdict", ""))
            spurious_count = sum(1 for e in EVALUATIONS if "Spurious" in e.get("evaluation", {}).get("verdict", ""))
            scores = [e.get("evaluation", {}).get("triad_quality_score", 0) for e in EVALUATIONS if "triad_quality_score" in e.get("evaluation", {})]
            avg_score = round(sum(scores) / max(len(scores), 1), 2)

            return self.send_json({
                "n_papers": 42164,
                "n_edges": 166993,
                "n_simplices_total": 94180,
                "n_ranked_triads": len(TRIADS_1000),
                "n_evaluations": len(EVALUATIONS),
                "n_hypotheses": len(EVALUATIONS) * 5,
                "avg_synergy_score": avg_score,
                "verdict_breakdown": {
                    "genuine_synergy": synergy_count,
                    "moderate_overlap": overlap_count,
                    "spurious": spurious_count
                }
            })

        if path == "/api/evaluations":
            verdict_filter = params.get("verdict", [None])[0]
            search_query = params.get("q", [""])[0].lower()

            results = COMBINED_EVALS
            if verdict_filter and verdict_filter != "all":
                results = [
                    e for e in results 
                    if verdict_filter.lower() in e.get("evaluation", {}).get("verdict", "").lower()
                ]

            if search_query:
                filtered = []
                for e in results:
                    text_blob = json.dumps(e, ensure_ascii=False).lower()
                    if search_query in text_blob:
                        filtered.append(e)
                results = filtered

            return self.send_json({"evaluations": results, "total": len(results)})

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
                    t for t in filtered
                    if query in t["title_u"].lower() or query in t["title_v"].lower() or query in t["title_w"].lower()
                    or query in t["paper_id_u"] or query in t["paper_id_v"] or query in t["paper_id_w"]
                ]

            total = len(filtered)
            start = (page - 1) * limit
            end = start + limit
            page_items = filtered[start:end]

            return self.send_json({
                "page": page,
                "limit": limit,
                "total": total,
                "total_pages": (total + limit - 1) // limit,
                "triads": page_items
            })

        if path.startswith("/api/triad/"):
            try:
                tid = int(path.split("/")[-1])
                # Find in triads
                triad_csv = next((t for t in TRIADS_1000 if t["rank"] == tid), None)
                digest = DIGESTS_BY_RANK.get(tid)
                eval_obj = next((e for e in EVALUATIONS if e.get("triad_id") == tid), None)
                return self.send_json({
                    "triad": triad_csv,
                    "digest": digest,
                    "evaluation": eval_obj
                })
            except Exception as e:
                return self.send_json({"error": str(e)}, status=HTTPStatus.NOT_FOUND)

        # Default static files
        if path == "/" or path == "":
            self.path = "/index.html"

        return super().do_GET()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Paper Simplicial Discovery Web App")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8501, help="Port to bind (default: 8501)")
    args = parser.parse_args()

    server_address = (args.host, args.port)
    httpd = ThreadingHTTPServer(server_address, AppHandler)
    host_display = "localhost" if args.host == "0.0.0.0" else args.host
    print(f"\n==================================================================")
    print(f"🚀 Paper Simplicial Discovery Web App is LIVE!")
    print(f"   URL: http://{host_display}:{args.port}")
    print(f"   Serving from: {WEBAPP_DIR}")
    print(f"==================================================================\n", flush=True)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web app...")
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
