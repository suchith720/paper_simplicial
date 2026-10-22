"""English-ness signals for papers (corpus lang scores + title heuristics)."""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np

from project_paths import ROOT

LANG_SCORES_PATH = ROOT.parent / "lang_scores_510k.npz"

EN_WORDS = frozenset(
    "the of and to in is that for with on as are by this be was were it from at "
    "which or we these has have not can been their its".split()
)
_WORD_RE = re.compile(r"[a-z]+")


def title_latin_share(title: str) -> float:
    letters = latin = 0
    for c in title or "":
        if c.isalpha():
            letters += 1
            if c < "\x80" or unicodedata.name(c, "").startswith("LATIN"):
                latin += 1
    return latin / max(letters, 1)


def title_function_word_ratio(title: str) -> float:
    words = _WORD_RE.findall((title or "").lower())
    if not words:
        return 0.0
    return sum(w in EN_WORDS for w in words) / len(words)


def title_englishness(title: str) -> float:
    """Combined 0–1 score: Latin script in title + common English function words."""
    latin = title_latin_share(title)
    fw = title_function_word_ratio(title)
    return 0.45 * latin + 0.55 * fw


@lru_cache(maxsize=1)
def load_lang_scores(path: Path | None = None) -> dict[str, np.ndarray]:
    p = path or LANG_SCORES_PATH
    if not p.is_file():
        raise FileNotFoundError(f"Language scores not found: {p}")
    z = np.load(p)
    return {"latin": np.asarray(z["latin"], dtype=np.float32), "en": np.asarray(z["en"], dtype=np.float32)}


def corpus_english_scores(row_idx: np.ndarray, *, lang_path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    lang = load_lang_scores(lang_path)
    rows = np.asarray(row_idx, dtype=np.int64)
    return lang["en"][rows], lang["latin"][rows]
