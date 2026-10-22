"""Load stacked MatSciBERT embeddings from embeddings_en/chunk_*.npz."""
from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PaperBatch:
    row_idx: np.ndarray
    paper_id: np.ndarray
    title: np.ndarray
    head: np.ndarray
    windows_mean: np.ndarray
    stacked: np.ndarray


def stacked_features(head: np.ndarray, windows_mean: np.ndarray) -> np.ndarray:
    return np.concatenate([head, windows_mean], axis=-1).astype(np.float32, copy=False)


def list_chunk_files(embeddings_dir: Path) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(embeddings_dir / "chunk_*.npz")))


def load_chunks_subset(
    chunk_files: list[Path],
    local_indices: np.ndarray,
    chunk_ids: np.ndarray,
) -> PaperBatch:
    """Load selected papers from multiple chunks (chunk_ids[i], local_indices[i])."""
    if len(local_indices) != len(chunk_ids):
        raise ValueError("local_indices and chunk_ids must have the same length")
    n = len(local_indices)
    if n == 0:
        raise ValueError("empty subset")

    by_chunk: dict[int, list[tuple[int, int]]] = {}
    for out_i, (cid, li) in enumerate(zip(chunk_ids, local_indices, strict=True)):
        by_chunk.setdefault(int(cid), []).append((int(li), out_i))

    dim_head = dim_win = None
    head = win = None
    row_idx = np.empty(n, dtype=np.int64)
    paper_id = np.empty(n, dtype=object)
    title = np.empty(n, dtype=object)

    for cid, pairs in sorted(by_chunk.items()):
        path = chunk_files[cid]
        z = np.load(path, allow_pickle=True)
        if dim_head is None:
            dim_head = int(z["head"].shape[1])
            dim_win = int(z["windows_mean"].shape[1])
            head = np.empty((n, dim_head), dtype=np.float32)
            win = np.empty((n, dim_win), dtype=np.float32)
        loc = np.array([p[0] for p in pairs], dtype=np.int64)
        out_pos = np.array([p[1] for p in pairs], dtype=np.int64)
        head[out_pos] = z["head"][loc]
        win[out_pos] = z["windows_mean"][loc]
        row_idx[out_pos] = z["row_idx"][loc]
        paper_id[out_pos] = z["paper_id"][loc]
        title[out_pos] = z["title"][loc]

    assert head is not None and win is not None
    return PaperBatch(
        row_idx=row_idx,
        paper_id=paper_id,
        title=title,
        head=head,
        windows_mean=win,
        stacked=stacked_features(head, win),
    )


def load_manifest_batch(manifest_path: Path, *, embeddings_dir: Path) -> PaperBatch:
    """Load stacked embeddings for papers listed in a manifest.npz file."""
    m = np.load(manifest_path, allow_pickle=True)
    chunk_files = list_chunk_files(embeddings_dir)
    return load_chunks_subset(
        chunk_files,
        np.asarray(m["chunk_local_idx"], dtype=np.int64),
        np.asarray(m["chunk_id"], dtype=np.int16),
    )


def load_chunk(path: Path) -> PaperBatch:
    z = np.load(path, allow_pickle=True)
    head = np.asarray(z["head"], dtype=np.float32)
    windows_mean = np.asarray(z["windows_mean"], dtype=np.float32)
    return PaperBatch(
        row_idx=np.asarray(z["row_idx"], dtype=np.int64),
        paper_id=np.asarray(z["paper_id"]),
        title=np.asarray(z["title"]),
        head=head,
        windows_mean=windows_mean,
        stacked=stacked_features(head, windows_mean),
    )
