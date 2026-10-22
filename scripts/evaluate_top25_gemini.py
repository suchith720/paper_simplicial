import json
import os
import re
import subprocess
import time
from pathlib import Path

BASE_DIR = Path('/home/mridul/home2/elseview/paper_simplicial/extracted_simplicial/run_english_science/complexes/review_top25')
EVAL_DIR = BASE_DIR / 'evaluations'
EVAL_DIR.mkdir(parents=True, exist_ok=True)

DIGEST_PATH = BASE_DIR / 'top25_triads_executive_digest.json'
with open(DIGEST_PATH) as f:
    digest_data = json.load(f)

triads = digest_data['triads']

PROMPT_TEMPLATE = """You are a senior principal investigator, scientific reviewer, and interdisciplinary research strategist. 

You are evaluating a candidate scientific paper triad—three research papers discovered as a 2-simplex (triangle) in a topological simplicial complex lifted from a kNN embedding graph. You are provided with each paper's Title, Dataset ID, and an Executive Scientific Digest structured into:
1. Abstract & Problem Formulation
2. Introduction & Research Motivation
3. Core Methodology & Architecture
4. Findings, Conclusions & Future Work

Pairwise embedding cosine similarities are also provided.

### YOUR OBJECTIVE:
1. Critical Triad Evaluation & Rating:
   - Critically evaluate whether this triad represents a genuine, deep scientific synergy or merely a surface-level semantic coincidence (e.g. sharing generic buzzwords like "optimization", "simulation", or "modeling").
   - Synthesize the complementary mechanisms, shared methodologies, or physical analogies across the three papers.
   - Assign an overall Triad Quality Score from 1.0 to 10.0 (1.0 = spurious/unrelated, 5.0 = standard domain overlap, 10.0 = transformative breakthrough potential).
   - Provide a detailed, rigorous justification for your rating citing specific technical elements from the digests.

2. Formulate 5 High-Impact Research Statements:
   - Propose exactly 5 distinct, concrete, testable, and methodologically sound research statements (hypotheses or project directions) that directly bridge the methodologies, materials, mechanisms, or domain problems of ALL THREE PAPERS.
   - Do NOT propose trivial juxtapositions. Synthesize higher-order conceptual bridges leveraging the complementary strengths of the triad.

3. Impact Reasoning for Each Statement:
   - Explain the Scientific Breakthrough, Bottleneck Resolved, Real-World Application, and Estimated Feasibility (High, Medium, or High-Risk/High-Reward).

### INPUT TRIAD DATA:
{triad_json}

### REQUIRED OUTPUT FORMAT:
Respond ONLY with a single valid JSON object strictly matching this schema (no preamble or commentary):
{{
  "triad_id": {rank},
  "evaluation": {{
    "triad_quality_score": 8.5,
    "verdict": "Genuine Cross-Disciplinary Synergy" | "Moderate Methodological Overlap" | "Spurious / Generic Co-occurrence",
    "dimension_ratings": {{
      "thematic_coherence": 8.0,
      "methodological_complementarity": 9.0,
      "non_triviality": 8.5
    }},
    "detailed_reasoning": "..."
  }},
  "research_statements": [
    {{
      "statement_id": 1,
      "title": "...",
      "research_statement": "...",
      "mechanistic_bridge": "...",
      "impact_analysis": {{
        "scientific_breakthrough": "...",
        "bottleneck_resolved": "...",
        "real_world_application": "..."
      }},
      "estimated_feasibility": "High" | "Medium" | "High-Risk/High-Reward"
    }},
    {{ "statement_id": 2, "title": "...", "research_statement": "...", "mechanistic_bridge": "...", "impact_analysis": {{ "scientific_breakthrough": "...", "bottleneck_resolved": "...", "real_world_application": "..." }}, "estimated_feasibility": "..." }},
    {{ "statement_id": 3, "title": "...", "research_statement": "...", "mechanistic_bridge": "...", "impact_analysis": {{ "scientific_breakthrough": "...", "bottleneck_resolved": "...", "real_world_application": "..." }}, "estimated_feasibility": "..." }},
    {{ "statement_id": 4, "title": "...", "research_statement": "...", "mechanistic_bridge": "...", "impact_analysis": {{ "scientific_breakthrough": "...", "bottleneck_resolved": "...", "real_world_application": "..." }}, "estimated_feasibility": "..." }},
    {{ "statement_id": 5, "title": "...", "research_statement": "...", "mechanistic_bridge": "...", "impact_analysis": {{ "scientific_breakthrough": "...", "bottleneck_resolved": "...", "real_world_application": "..." }}, "estimated_feasibility": "..." }}
  ]
}}
"""

def extract_json(raw: str) -> dict:
    # Remove code blocks if present
    text = raw.strip()
    m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if m:
        text = m.group(1).strip()
    return json.loads(text)

def evaluate_triad(triad: dict) -> dict:
    rank = triad['rank']
    prompt_content = PROMPT_TEMPLATE.format(
        triad_json=json.dumps(triad, indent=2, ensure_ascii=False),
        rank=rank
    )
    
    cmd = [
        '/home/mridul/.local/bin/agy',
        '--model', 'gemini-3.8-flash-high',
        '-p', prompt_content
    ]
    
    for attempt in range(4):
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
            if res.returncode != 0:
                print(f"  Attempt {attempt+1} failed with error: {res.stderr[:200]}")
                time.sleep(5)
                continue
            
            parsed = extract_json(res.stdout)
            return parsed
        except Exception as e:
            print(f"  Attempt {attempt+1} exception: {e}")
            time.sleep(5)
            
    print(f"ERROR: Could not evaluate triad {rank} after 4 attempts.")
    return None

print(f"Starting Gemini evaluation for {len(triads)} triads...")
all_evals = []

for idx, triad in enumerate(triads, 1):
    rank = triad['rank']
    eval_file = EVAL_DIR / f"triad_{rank:02d}_eval.json"
    
    if eval_file.exists():
        try:
            with open(eval_file) as f:
                data = json.load(f)
            print(f"[{idx}/{len(triads)}] Triad #{rank} already evaluated (Score: {data['evaluation']['triad_quality_score']}, Verdict: {data['evaluation']['verdict']})")
            all_evals.append(data)
            continue
        except Exception:
            pass

    print(f"[{idx}/{len(triads)}] Evaluating Triad #{rank}...", flush=True)
    t0 = time.time()
    eval_result = evaluate_triad(triad)
    elapsed = time.time() - t0
    
    if eval_result is not None:
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(eval_result, f, indent=2, ensure_ascii=False)
            
        score = eval_result.get('evaluation', {}).get('triad_quality_score', 'N/A')
        verdict = eval_result.get('evaluation', {}).get('verdict', 'N/A')
        print(f"  Completed in {elapsed:.1f}s -> Score: {score}/10 | Verdict: {verdict}", flush=True)
        all_evals.append(eval_result)
    else:
        print(f"  FAILED to evaluate Triad #{rank}. Skipping.", flush=True)
    time.sleep(1)

# Save Master Evaluation
master_eval_path = EVAL_DIR / 'top25_evaluations_master.json'
with open(master_eval_path, 'w', encoding='utf-8') as f:
    json.dump({
        'n_evaluations': len(all_evals),
        'evaluator_model': 'gemini-3.8-flash-high',
        'evaluations': all_evals
    }, f, indent=2, ensure_ascii=False)

# Generate Summary Report
summary_md = EVAL_DIR / 'EVALUATION_REPORT.md'
triads_by_rank = {t['rank']: t for t in digest_data.get('triads', [])}

def clean_title(s):
    return ' '.join(s.replace(r'\ud', '').replace(r'\u27', "'").split())

with open(summary_md, 'w', encoding='utf-8') as f:
    f.write("# Top 25 Paper Triads: Gemini Scientific Evaluation & Research Synthesis\n\n")
    f.write(f"Evaluated **{len(all_evals)} candidate simplicial triads** using Gemini with high-reasoning effort.\n\n")
    f.write("| Rank | Quality Score | Verdict | Paper 1 Title | Paper 2 Title | Paper 3 Title | Top Synthesized Research Direction | Thematic Coherence | Complementarity | Non-Triviality |\n")
    f.write("| :---: | :---: | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |\n")
    for ev in all_evals:
        t_id = ev.get('triad_id', 'N/A')
        e = ev.get('evaluation', {})
        score = e.get('triad_quality_score', 'N/A')
        verdict = e.get('verdict', 'N/A')
        dim = e.get('dimension_ratings', {})
        tc = dim.get('thematic_coherence', 'N/A')
        mc = dim.get('methodological_complementarity', 'N/A')
        nt = dim.get('non_triviality', 'N/A')
        top_title = ev.get('research_statements', [{}])[0].get('title', 'N/A')

        triad = triads_by_rank.get(t_id, {})
        p1 = clean_title(triad.get('papers', {}).get('paper_1', {}).get('title', 'N/A'))
        p2 = clean_title(triad.get('papers', {}).get('paper_2', {}).get('title', 'N/A'))
        p3 = clean_title(triad.get('papers', {}).get('paper_3', {}).get('title', 'N/A'))

        f.write(f"| **#{t_id}** | **{score}/10** | {verdict} | *{p1}* | *{p2}* | *{p3}* | *{top_title}* | {tc} | {mc} | {nt} |\n")

print("\nEvaluation successfully completed for all 25 triads!")
print(f"Master evaluation JSON: {master_eval_path}")
print(f"Summary Report: {summary_md}")
