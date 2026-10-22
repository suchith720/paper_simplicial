#!/usr/bin/env python3
"""Rank triangles and export top hypotheses (paper titles) as JSON + CSV."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import EXTRACTED_DIR  # noqa: E402
from lib.triangle_rank import rank_triangles  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=EXTRACTED_DIR / "run_10k")
    p.add_argument("--method", choices=("clique", "neighbor"), default="clique")
    p.add_argument(
        "--score",
        choices=("min", "mean", "mean_min", "geom", "geom_min", "english_tight"),
        default="mean_min",
    )
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--min-corpus-en", type=float, default=0.0, help="Min corpus English score on all 3 papers")
    p.add_argument("--min-title-en", type=float, default=0.0, help="Min title Englishness on all 3 papers")
    p.add_argument("--min-title-latin", type=float, default=0.0, help="Min Latin-letter share in all 3 titles")
    args = p.parse_args()

    rec_path = args.run_dir / "complexes" / args.method / "complex.pkl"
    record = pickle.load(open(rec_path, "rb"))["record"]
    n_tri = len(record["triangles"])
    ranked = rank_triangles(
        record,
        score_mode=args.score,
        min_corpus_en=args.min_corpus_en,
        min_title_en=args.min_title_en,
        min_title_latin=args.min_title_latin,
        english_only=(args.score == "english_tight"),
    )
    top = ranked[: args.top]

    out_dir = args.run_dir / "hypotheses"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.method}_{args.score}_top{args.top}"

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(
            {
                "score_mode": args.score,
                "description": (
                    "Higher score = tighter semantic triple on kNN edge cosines. "
                    "english_tight = filter English-looking titles + corpus en, then geom_min."
                ),
                "n_triangles_total": n_tri,
                "n_triangles_after_filter": len(ranked),
                "triangles": [asdict(t) for t in top],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = out_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            ["rank", "score", "w_uv", "w_vw", "w_uw", "title_u", "title_v", "title_w", "paper_id_u", "paper_id_v", "paper_id_w"]
        )
        for t in top:
            nu, nv, nw = t.nodes
            wu, wv, ww = t.edge_weights
            tu, tv, tw = t.titles
            pu, pv, pw = t.paper_ids
            writer.writerow([t.rank, f"{t.score:.6f}", wu, wv, ww, tu, tv, tw, pu, pv, pw])

    print(f"Triangles: {n_tri} total, {len(ranked)} after filters; wrote top {len(top)} to:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print("\nTop 5:")
    for t in top[:5]:
        print(f"  #{t.rank} score={t.score:.4f}")
        for title in t.titles:
            print(f"      · {title[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
