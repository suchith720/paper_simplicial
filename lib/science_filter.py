"""Identify English-language science papers in the embedded CORE corpus."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

from lib.language import corpus_english_scores, title_englishness, title_latin_share

FilterMode = Literal["balanced", "strict"]

# Obvious STEM / quantitative science signals in titles.
_STEM = re.compile(
    r"\b("
    r"cell|cells|gene|genome|protein|molecular|enzyme|dna|rna|microbi|bacteria|virus|"
    r"species|ecolog|evolution|phylogen|marine|fisheries|"
    r"cancer|clinical|patient|diagnosis|therapy|surgery|hospital|pharmac|epidemiolog|biomarker|"
    r"quantum|magnetic|laser|optics|spectroscop|photon|plasma|"
    r"polymer|catalyst|nanoparticle|alloy|ceramic|electrolyte|battery|"
    r"algorithm|neural|machine learning|deep learning|software|database|"
    r"climate|carbon|seismic|hydrolog|atmospher|"
    r"engineer|robot|sensor|circuit|aerospace|"
    r"theorem|stochastic|bayesian|regression|optimization|"
    r"crystal|diffraction|chromatograph|microscop"
    r")\b",
    re.I,
)

# Strong non-STEM humanities / law / regional-studies signals (title-only).
_NON_SCIENCE = re.compile(
    r"\b("
    r"law|legal|court|jurisdiction|attorney|bill of rights|parliamentary|litigation|"
    r"museum|heritage|jazz|music|poetry|literature|philosoph|theatre|theater|"
    r"te papa|māori|maori|wellington jazz|"
    r"mediation in the pacific|performing the hyphen"
    r")\b",
    re.I,
)

# Weaker science-methodology cues (used only when no explicit NON_SCIENCE hit).
_METHOD = re.compile(
    r"\b("
    r"experimental|simulation|computational|numerical|measurement|"
    r"systematic review|randomized|rct|meta-analysis|"
    r"mechanism|synthesis|characterization"
    r")\b",
    re.I,
)


@dataclass(frozen=True)
class ScienceFilterConfig:
    mode: FilterMode = "balanced"
    min_corpus_en: float = 0.20
    min_corpus_latin: float = 0.90
    min_title_en: float = 0.12
    min_title_latin: float = 0.92


def passes_english_gates(
    title: str,
    *,
    corpus_en: float,
    corpus_latin: float,
    cfg: ScienceFilterConfig,
) -> bool:
    if corpus_en < cfg.min_corpus_en or corpus_latin < cfg.min_corpus_latin:
        return False
    if title_englishness(title) < cfg.min_title_en:
        return False
    if title_latin_share(title) < cfg.min_title_latin:
        return False
    return True


def _has_match(rx: re.Pattern[str], text: str) -> bool:
    m = rx.search(text)
    return m is not None and m.end() > m.start()


def is_science_title(title: str, *, mode: FilterMode = "balanced") -> bool:
    t = title or ""
    stem = _has_match(_STEM, t)
    non = _has_match(_NON_SCIENCE, t)
    if mode == "strict":
        return stem and not non
    if non and not stem:
        return False
    if stem:
        return True
    return _has_match(_METHOD, t)


def is_english_science_paper(
    title: str,
    *,
    corpus_en: float,
    corpus_latin: float,
    cfg: ScienceFilterConfig | None = None,
) -> bool:
    cfg = cfg or ScienceFilterConfig()
    if not passes_english_gates(title, corpus_en=corpus_en, corpus_latin=corpus_latin, cfg=cfg):
        return False
    return is_science_title(title, mode=cfg.mode)


def mask_english_science(
    titles: np.ndarray,
    row_idx: np.ndarray,
    *,
    cfg: ScienceFilterConfig | None = None,
) -> np.ndarray:
    cfg = cfg or ScienceFilterConfig()
    rows = np.asarray(row_idx, dtype=np.int64)
    corpus_en, corpus_latin = corpus_english_scores(rows)
    out = np.zeros(len(titles), dtype=bool)
    for i, title in enumerate(titles):
        out[i] = is_english_science_paper(
            str(title),
            corpus_en=float(corpus_en[i]),
            corpus_latin=float(corpus_latin[i]),
            cfg=cfg,
        )
    return out
