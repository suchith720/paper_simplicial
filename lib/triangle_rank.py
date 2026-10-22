"""Score 2-simplices (triangles) for hypothesis-style ranking."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib.language import corpus_english_scores, title_englishness, title_latin_share


@dataclass(frozen=True)
class RankedTriangle:
    rank: int
    score: float
    nodes: tuple[int, int, int]
    edge_weights: tuple[float, float, float]
    titles: tuple[str, str, str]
    paper_ids: tuple[str, str, str]
    row_idx: tuple[int, int, int]
    min_edge: float = 0.0
    mean_edge: float = 0.0
    min_corpus_en: float = 0.0
    min_title_en: float = 0.0


def edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def triangle_score(weights: tuple[float, float, float], mode: str) -> float:
    a, b, c = weights
    if mode == "min":
        return float(min(a, b, c))
    if mode == "mean":
        return float((a + b + c) / 3.0)
    if mode == "mean_min":
        return float(min(a, b, c) * (a + b + c) / 3.0)
    if mode == "geom":
        return float((a * b * c) ** (1.0 / 3.0))
    if mode == "geom_min":
        m = min(a, b, c)
        return float(m * (a * b * c) ** (1.0 / 3.0))
    raise ValueError(f"Unknown score mode: {mode!r}")


def rank_triangles(
    record: dict,
    *,
    score_mode: str = "mean_min",
    lang_path: Path | None = None,
    min_corpus_en: float = 0.0,
    min_title_en: float = 0.0,
    min_title_latin: float = 0.0,
    english_only: bool = False,
) -> list[RankedTriangle]:
    titles = list(record["title"])
    paper_ids = list(record["paper_id"])
    row_idx_list = list(record["row_idx"])
    row_arr = np.asarray(row_idx_list, dtype=np.int64)
    corpus_en, corpus_latin = corpus_english_scores(row_arr, lang_path=lang_path)
    title_en = np.array([title_englishness(t) for t in titles], dtype=np.float32)
    title_latin = np.array([title_latin_share(t) for t in titles], dtype=np.float32)

    wmap = {edge_key(int(e[0]), int(e[1])): float(w) for e, w in zip(record["edges"], record["edge_weights"])}

    tight_mode = score_mode
    if score_mode == "english_tight":
        tight_mode = "geom_min"
        english_only = True
        if min_corpus_en <= 0.0:
            min_corpus_en = 0.20
        if min_title_en <= 0.0:
            min_title_en = 0.12
        if min_title_latin <= 0.0:
            min_title_latin = 0.92

    scored: list[tuple[float, tuple[int, int, int], tuple[float, float, float], float, float, float, float]] = []
    for t in record["triangles"]:
        u, v, w = int(t[0]), int(t[1]), int(t[2])
        nodes = (u, v, w)
        ws = (
            wmap.get(edge_key(u, v), 0.0),
            wmap.get(edge_key(v, w), 0.0),
            wmap.get(edge_key(u, w), 0.0),
        )
        mc = float(min(corpus_en[u], corpus_en[v], corpus_en[w]))
        mt = float(min(title_en[u], title_en[v], title_en[w]))
        ml = float(min(title_latin[u], title_latin[v], title_latin[w]))
        if english_only or min_corpus_en > 0 or min_title_en > 0 or min_title_latin > 0:
            if mc < min_corpus_en or mt < min_title_en or ml < min_title_latin:
                continue
        sc = triangle_score(ws, tight_mode)
        scored.append((sc, nodes, ws, min(ws), sum(ws) / 3.0, mc, mt))

    scored.sort(key=lambda x: (-x[0], x[1]))
    out: list[RankedTriangle] = []
    for i, (sc, nodes, ws, mn, mean_e, mc, mt) in enumerate(scored, start=1):
        u, v, w = nodes
        out.append(
            RankedTriangle(
                rank=i,
                score=sc,
                nodes=nodes,
                edge_weights=ws,
                titles=(titles[u], titles[v], titles[w]),
                paper_ids=(str(paper_ids[u]), str(paper_ids[v]), str(paper_ids[w])),
                row_idx=(int(row_idx_list[u]), int(row_idx_list[v]), int(row_idx_list[w])),
                min_edge=mn,
                mean_edge=mean_e,
                min_corpus_en=mc,
                min_title_en=mt,
            )
        )
    return out
