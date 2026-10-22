#!/usr/bin/env python3
"""Scan all embedding chunks; save English science paper ids and indices."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.embeddings import list_chunk_files  # noqa: E402
from lib.io import write_json  # noqa: E402
from lib.science_filter import ScienceFilterConfig, mask_english_science  # noqa: E402
from project_paths import EMBEDDINGS_DIR, ENGLISH_SCIENCE_MANIFEST  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings-dir", type=Path, default=EMBEDDINGS_DIR)
    p.add_argument("--out", type=Path, default=ENGLISH_SCIENCE_MANIFEST)
    p.add_argument("--mode", choices=("balanced", "strict"), default="balanced")
    p.add_argument("--min-corpus-en", type=float, default=0.20)
    args = p.parse_args()

    chunk_files = list_chunk_files(args.embeddings_dir)
    if not chunk_files:
        print(f"No chunks in {args.embeddings_dir}", file=sys.stderr)
        return 1

    cfg = ScienceFilterConfig(mode=args.mode, min_corpus_en=args.min_corpus_en)
    global_idx_parts: list[np.ndarray] = []
    row_idx_parts: list[np.ndarray] = []
    paper_id_parts: list[np.ndarray] = []
    title_parts: list[np.ndarray] = []
    chunk_id_parts: list[np.ndarray] = []
    local_idx_parts: list[np.ndarray] = []

    global_offset = 0
    n_total = 0
    n_sel = 0

    for cid, cpath in enumerate(chunk_files):
        z = np.load(cpath, allow_pickle=True)
        titles = z["title"]
        row_idx = np.asarray(z["row_idx"], dtype=np.int64)
        paper_id = np.asarray(z["paper_id"])
        n = len(row_idx)
        n_total += n

        keep = mask_english_science(titles, row_idx, cfg=cfg)
        n_sel += int(keep.sum())
        if not keep.any():
            global_offset += n
            continue

        loc = np.flatnonzero(keep)
        global_idx_parts.append(global_offset + loc.astype(np.int64))
        row_idx_parts.append(row_idx[keep])
        paper_id_parts.append(paper_id[keep])
        title_parts.append(titles[keep])
        chunk_id_parts.append(np.full(loc.shape[0], cid, dtype=np.int16))
        local_idx_parts.append(loc.astype(np.int32))
        global_offset += n
        print(f"  {cpath.name}: {int(keep.sum()):5d} / {n:5d} selected", flush=True)

    global_idx = np.concatenate(global_idx_parts)
    row_idx_out = np.concatenate(row_idx_parts)
    paper_id_out = np.concatenate(paper_id_parts)
    title_out = np.concatenate(title_parts)
    chunk_id_out = np.concatenate(chunk_id_parts)
    local_idx_out = np.concatenate(local_idx_parts)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        global_idx=global_idx,
        row_idx=row_idx_out,
        paper_id=paper_id_out,
        title=title_out,
        chunk_id=chunk_id_out,
        chunk_local_idx=local_idx_out,
        embeddings_dir=np.array([str(args.embeddings_dir.resolve())]),
        filter_mode=np.array([cfg.mode]),
    )

    meta = {
        "n_corpus_english_embedded": n_total,
        "n_english_science_selected": int(len(global_idx)),
        "fraction_selected": round(len(global_idx) / max(n_total, 1), 6),
        "n_chunks": len(chunk_files),
        "filter": {
            "mode": cfg.mode,
            "min_corpus_en": cfg.min_corpus_en,
            "min_corpus_latin": cfg.min_corpus_latin,
            "min_title_en": cfg.min_title_en,
            "min_title_latin": cfg.min_title_latin,
        },
        "chunk_files": [str(p) for p in chunk_files],
        "manifest_npz": str(out),
    }
    write_json(out.with_suffix(".json"), meta)

    print(
        f"\nSelected {len(global_idx):,} / {n_total:,} English-embedded papers "
        f"({100 * len(global_idx) / n_total:.2f}%)"
    )
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
