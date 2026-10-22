"""Save extracted simplicial complexes under extracted_simplicial/."""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

RECORD_VERSION = 1


@dataclass(frozen=True)
class PaperSimplicialRecord:
    method: str
    n_nodes: int
    n_edges_graph: int
    edges: tuple[tuple[int, int], ...]
    edge_weights: tuple[float, ...]
    triangles: tuple[tuple[int, int, int], ...]
    row_idx: tuple[int, ...]
    paper_id: tuple[str, ...]
    title: tuple[str, ...]
    vr_radius: float | None = None


def save_simplicial_record(record: PaperSimplicialRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": RECORD_VERSION, "record": asdict(record)}
    with path.open("wb") as fp:
        pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)


def save_graph_pickle(
    path: Path,
    *,
    edges: list[tuple[int, int, float]],
    row_idx: np.ndarray,
    paper_id: np.ndarray,
    title: np.ndarray,
    centering_mean: np.ndarray,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": RECORD_VERSION,
        "meta": meta,
        "centering_mean": centering_mean,
        "row_idx": np.asarray(row_idx, dtype=np.int64),
        "paper_id": np.asarray(paper_id),
        "title": np.asarray(title),
        "edges": edges,
    }
    with path.open("wb") as fp:
        pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2)
