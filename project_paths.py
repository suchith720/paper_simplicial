"""Paths for the Paper Simplicial Discovery repository."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

LIB_DIR = ROOT / "lib"
SCRIPTS_DIR = ROOT / "scripts"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
PROMPTS_DIR = ROOT / "prompts"

# Key results
TOP_1000_CSV = RESULTS_DIR / "top_1000_triangles.csv"
REVIEW_TOP25_DIR = RESULTS_DIR / "review_top25"
EXECUTIVE_DIGEST_JSON = REVIEW_TOP25_DIR / "top25_triads_executive_digest.json"
READABLE_ABSTRACTS_JSON = REVIEW_TOP25_DIR / "top25_triads_readable_abstracts.json"
EVALUATIONS_DIR = REVIEW_TOP25_DIR / "evaluations"

# External upstream data (raw heavy chunks and manifests)
EXTERNAL_EMBEDDINGS_DIR = Path("/DATATWO/users/mridul2/elseview/paper_simplicial/embeddings_en")
EXTERNAL_EXTRACTED_DIR = Path("/DATATWO/users/mridul2/elseview/paper_simplicial/extracted_simplicial")
EXTERNAL_MANIFEST = EXTERNAL_EXTRACTED_DIR / "english_science" / "manifest.npz"
EXTERNAL_RUN_DIR = EXTERNAL_EXTRACTED_DIR / "run_english_science"
