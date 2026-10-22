# Scientific Paper Simplicial Discovery

A topological deep learning and LLM-assisted pipeline for discovering higher-order scientific interactions and formulating cross-disciplinary research hypotheses from scientific literature.

---

## 1. Executive Summary

Standard natural language processing and graph retrieval methods rely strictly on **pairwise** relations ($A \sim B$, $B \sim C$). In dense embedding spaces, pairwise similarities are prone to semantic drift and intransitivity (just because Paper A is similar to Paper B, and Paper B is similar to Paper C, does not imply that A and C share a valid scientific connection).

This project lifts pairwise scientific paper embedding graphs into **topological simplicial complexes** (2-simplices / triangles). When three papers form a closed 3-clique in the $k\text{NN}$ graph, they form a **2-simplex** representing a mutual, multi-way scientific triad. By scoring these triads for semantic tightness and filtering for scientific rigor, the pipeline identifies high-synergy research opportunities and prompts **Gemini Pro** to critically review the triads and synthesize testable, high-impact research hypotheses.

---

## 2. Pipeline Architecture

```
                  Scientific Papers (CoreX Corpus)
                                 │
                                 ▼
                     MatSciBERT Mean-Pooled Embeddings
                                 │
                                 ▼
                    Deduplication (Cosine ≥ 0.99)
                  45,666 raw papers ➔ 42,164 unique
                                 │
                                 ▼
                   Undirected kNN Graph (k=5)
                   |V| = 42,164  |  |E| = 166,993
                                 │
                                 ▼
                     TopoNetX Simplicial Lifting
               (Clique Complex & Neighbor Complex Lift)
                       |T| = 94,180 Triangles
                                 │
                                 ▼
                 Triad Scoring & Tightness Ranking
                     (Top 1,000 Triads Ranked)
                                 │
                                 ▼
            Executive Scientific Digest Extraction (Top 25)
      [Abstract + Problem Aims + Methodology + Findings/Future Work]
                                 │
                                 ▼
                Gemini Pro Scientific Evaluation
       [Quality Score /10 + Verdict + 5 Testable Research Statements]
```

---

## 3. Repository Organization

```
paper_simplicial/
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Lightweight Python dependencies
├── project_paths.py               # Path configurations
├── .gitignore                     # Git ignore rules (prunes heavy weights and caches)
│
├── lib/                           # Core mathematical and processing modules
│   ├── graph.py                   # PyTorch GPU kNN graph construction with union symmetrization
│   ├── simplicial.py              # TopoNetX lifts (Clique complex, Neighbor complex, Vietoris-Rips)
│   ├── triangle_rank.py           # Multi-mode triangle scoring (min, geom, geom_min, english_tight)
│   ├── dedupe.py                  # Cosine threshold deduplication & cluster merging
│   ├── language.py                # Englishness filters, Latin script shares, function word ratios
│   ├── science_filter.py          # Academic science vocabulary and taxonomy filters
│   ├── embeddings.py              # Manifest loaders and feature stacking
│   └── io.py                      # Data records, graph serialization, and JSON helpers
│
├── scripts/                       # Executable pipeline and analysis scripts
│   ├── build_english_science_manifest.py  # Filters English science subset from corpus
│   ├── run_simplicial_from_manifest.py    # End-to-end pipeline: load -> dedupe -> kNN -> lift
│   ├── rank_top_triangles.py              # Scores and ranks all 94,180 triangles to CSV/JSON
│   ├── generate_executive_digest.py       # Extracts 4-section executive digests from raw papers
│   ├── evaluate_top25_gemini.py           # Gemini Pro automated peer-review and hypothesis generator
│   ├── export_top25_review.py             # Attaches unabridged full text for review
│   ├── visualize_simplicial.py            # Interactive 2D/3D simplicial complex visualizer
│   ├── visualize_simplicial_full_mesh.py  # High-resolution full mesh renderings
│   ├── visualize_sparse_top_triangles.py  # Renders top ranked triangles over graph
│   └── visualize_simplicial_labeled.py    # Local subgraphs with paper titles on nodes
│
├── prompts/                       # LLM prompt templates
│   └── gemini_pro_eval_prompt.md  # Detailed prompt with scoring rubric and JSON schema
│
├── results/                       # Important results and discoveries
│   ├── top_1000_triangles.csv     # Complete ranked list of top 1,000 triads across 94,180 simplices
│   ├── clique_english_tight_top5.json  # Baseline top 5 discovery JSON
│   ├── clique_english_tight_top5.csv   # Baseline top 5 discovery CSV
│   └── review_top25/              # In-depth package for top 25 triads
│       ├── README.md              # Master summary index table
│       ├── top25_triads_readable_abstracts.json  # Human-readable Title + Abstract JSON
│       ├── top25_triads_executive_digest.json    # Complete 4-section Executive Digest
│       ├── abstracts_json/        # Individual clean JSONs per triad
│       ├── executive_digest_json/ # Individual digest JSONs per triad
│       └── evaluations/           # Gemini Pro ratings and 5 research statements per triad
│
└── figures/                       # Representative figures and topological mesh metadata
    ├── clique_full_mesh_meta.json
    ├── clique_sparse_english_tight_top150_all_nodes.meta.json
    ├── clique_sparse_english_tight_top150_all_nodes.png
    ├── clique_sparse_english_tight_top400_all_nodes.meta.json
    ├── clique_sparse_english_tight_top400_all_nodes.png
    └── neighbor_full_mesh_meta.json
```

---

## 4. Key Discoveries from the Top Triads

The pipeline discovered tight scientific triads with edge cosine similarities $> 0.985$. The LLM evaluation revealed that the simplicial complex uncovers authentic, multi-scale interdisciplinary connections:

### Triad #1: Distributed Energy System Decarbonization Pipeline (Score: 7.8/10)
* **Paper 1**: *3D energy modeling optimisation using computational design software* (Urban building envelope & PV demand optimization).
* **Paper 2**: *Modelling and simulation of a microreactor for ethanol steam reforming* (Catalytic hydrogen conversion & storage).
* **Paper 3**: *Simulation and optimization of an HVDC network connecting offshore windfarms with the mainland* (HVDC transmission & Optimal Power Flow control).
* **Synthesized Breakthrough**: Grid-responsive sector coupling coordinating offshore wind surges through mainland HVDC lines with distributed catalytic hydrogen reforming to power 3D-optimized urban residential microgrids.

### Triad #2: Natural Products Total Synthesis & Catalysis (Score: 7.3/10)
* **Paper 1**: *Metal complexes as catalysts for the synthesis of heterocycles* (Pre-formed transition metal catalysts for click chemistry).
* **Paper 2**: *Investigation of novel thermal cyclisation reactions and studies on their application to the synthesis of selected natural products* (Thermal radical cyclizations for complex natural cores).
* **Paper 3**: *Studies towards the total synthesis of Herbimycin A* (Complex 19-membered lactam polyketide total synthesis).
* **Synthesized Breakthrough**: Transition-metal-templated metallo-radical macrocyclizations overcoming steric and conformational barriers in ansamycin antibiotic polyketides.

---

## 5. Setup & Usage

### 1. Installation

Create a Python $\ge 3.10$ environment and install dependencies:

```bash
conda create -n simplicial_env python=3.10 -y
conda activate simplicial_env
pip install -r requirements.txt
```

### 2. Triangle Tightness Ranking

To rank triangles from an extracted complex:

```bash
python scripts/rank_top_triangles.py \
    --run-dir extracted_simplicial/run_english_science \
    --method clique \
    --score english_tight \
    --top 1000
```

### 3. Extracting the Executive Scientific Digest

To extract the clean 4-section digest (Abstract, Introduction, Methodology, Future Work) from the raw corpus:

```bash
python scripts/generate_executive_digest.py
```

### 4. Running the Gemini Pro Evaluation Pipeline

To run the automated peer review and 5-hypothesis generation via the `agy` CLI:

```bash
python scripts/evaluate_top25_gemini.py
```

---

## 6. Citation & Acknowledgments

This framework integrates:
* **TopoNetX**: Topological domain representations and simplicial complex transformations.
* **MatSciBERT**: Materials and scientific literature contextual domain embeddings.
* **CoreX**: Multi-disciplinary open-access scientific publications dataset.
