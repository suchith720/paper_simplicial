from __future__ import annotations
import json
import re
from pathlib import Path
from datasets import load_dataset

out_dir = Path('/home/mridul/home2/elseview/paper_simplicial/extracted_simplicial/run_english_science/hypotheses/review_top25')
json_dir = out_dir / 'executive_digest_json'
out_dir.mkdir(parents=True, exist_ok=True)
json_dir.mkdir(parents=True, exist_ok=True)

hyp_json_path = '/home/mridul/home2/elseview/paper_simplicial/extracted_simplicial/run_english_science/hypotheses/clique_english_tight_top1000.json'
with open(hyp_json_path) as f:
    hyp_data = json.load(f)

triangles = hyp_data['triangles'][:25]
unique_rows = sorted(list({r for t in triangles for r in t['row_idx']}))

print(f'Loading dataset for {len(unique_rows)} unique papers...')
ds = load_dataset('scientifi-papers/scientific-papers', 'corex')['train']

def clean_spacing(t: str) -> str:
    parts = []
    for line in t.split('\n'):
        line = line.strip()
        if not line:
            continue
        prev = parts[-1] if parts else ""
        if prev.endswith("-") and len(prev) > 1 and prev[-2].isalpha() and line[0].islower():
            parts[-1] = prev[:-1] + line
        else:
            parts.append(line)
    return " ".join(parts)

def extract_sections(text: str) -> dict[str, str]:
    clean_lines = [l.strip() for l in text.split('\n') if l.strip()]
    full = '\n'.join(clean_lines)

    # 1. Abstract
    abstract = ''
    m_abs = list(re.finditer(r'(?:^|\n)\s*(?:abstract|summary)\b[:\s\-]*', full, re.IGNORECASE))
    for m in m_abs:
        cand = full[m.end():m.end()+3000].strip()
        lines = [l.strip() for l in cand.split('\n') if l.strip()]
        if any(re.search(r'\.{3,}|\bpage\b|\bchapter\b', l, re.IGNORECASE) for l in lines[:3]):
            continue
        coll = []
        for l in lines:
            if re.match(r'^(?:1[\.\s]+|chapter\s+1|introduction|resumen|keywords)\b', l, re.IGNORECASE) and len(coll) >= 3:
                break
            coll.append(l)
            if sum(len(x) for x in coll) > 2200:
                break
        abstract = clean_spacing(' '.join(coll))[:2000].strip()
        if len(abstract.split()) > 30:
            break

    # 2. Introduction / Motivation
    intro = ''
    m_intro = list(re.finditer(r'(?:^|\n)\s*(?:(?:1|chapter\s+1)[\.\s:]+)?introduction\b[:\s\-]*', full, re.IGNORECASE))
    for m in m_intro:
        cand = full[m.end():m.end()+4000].strip()
        lines = [l.strip() for l in cand.split('\n') if l.strip()]
        if any(re.search(r'\.{3,}', l) for l in lines[:3]):
            continue
        coll = []
        for l in lines:
            if re.match(r'^(?:2[\.\s]+|chapter\s+2|methods|state\s+of\s+the\s+art|background)\b', l, re.IGNORECASE) and len(coll) >= 5:
                break
            coll.append(l)
            if sum(len(x) for x in coll) > 3000:
                break
        intro = clean_spacing(' '.join(coll))[:2800].strip()
        if len(intro.split()) > 40:
            break

    # 3. Method / Architecture / Synthesis
    method = ''
    m_meth = list(re.finditer(r'(?:^|\n)\s*(?:(?:[2-4]|chapter\s+[2-4])[\.\s:]+)?(?:methodology|methods|experimental|system\s+model|mathematical\s+model|computational\s+method|synthesis)\b[:\s\-]*', full, re.IGNORECASE))
    for m in m_meth:
        cand = full[m.end():m.end()+4500].strip()
        lines = [l.strip() for l in cand.split('\n') if l.strip()]
        if any(re.search(r'\.{3,}', l) for l in lines[:3]):
            continue
        coll = []
        for l in lines:
            if re.match(r'^(?:(?:[3-6]|chapter\s+[3-6])[\.\s:]+)?(?:results|discussion|conclusions|validation)\b', l, re.IGNORECASE) and len(coll) >= 5:
                break
            coll.append(l)
            if sum(len(x) for x in coll) > 3500:
                break
        method = clean_spacing(' '.join(coll))[:3200].strip()
        if len(method.split()) > 40:
            break

    # 4. Conclusion / Future Work / Discussion
    concl = ''
    m_concl = list(re.finditer(r'(?:^|\n)\s*(?:(?:[5-8]|chapter\s+[5-8])[\.\s:]+)?(?:conclusion|conclusions|concluding\s+remarks|future\s+work|discussion\s+and\s+conclusion)\b[:\s\-]*', full, re.IGNORECASE))
    for m in m_concl:
        cand = full[m.end():m.end()+3500].strip()
        lines = [l.strip() for l in cand.split('\n') if l.strip()]
        if any(re.search(r'\.{3,}', l) for l in lines[:3]):
            continue
        coll = []
        for l in lines:
            if re.match(r'^(?:bibliography|references|acknowledgements|annex|appendix)\b', l, re.IGNORECASE) and len(coll) >= 3:
                break
            coll.append(l)
            if sum(len(x) for x in coll) > 3000:
                break
        concl = clean_spacing(' '.join(coll))[:2800].strip()
        if len(concl.split()) > 30:
            break

    return {
        'abstract_and_aims': abstract or clean_spacing(' '.join(clean_lines[:15]))[:1500],
        'introduction_and_motivation': intro or 'See abstract and methodology sections.',
        'core_methodology_and_architecture': method or 'See introduction and results sections.',
        'findings_and_conclusions': concl or 'See core results and discussion.'
    }

print('Extracting Executive Digest for all unique papers...')
papers_cache = {}
for r in unique_rows:
    p = ds[int(r)]
    raw_text = p['raw_fulltext'] or ''
    sections = extract_sections(raw_text)
    papers_cache[r] = {
        'paper_id': str(p['paper_id']),
        'title': str(p['title']),
        'total_chars_original': len(raw_text),
        'sections': sections
    }

triads_out = []
for idx, t in enumerate(triangles, 1):
    rank = t['rank']
    score = t['score']
    edge_weights = t['edge_weights']
    titles = t['titles']
    paper_ids = t['paper_ids']
    row_idxs = t['row_idx']

    papers_in_triad = []
    for r, pid, tit in zip(row_idxs, paper_ids, titles):
        p_data = papers_cache[r]
        papers_in_triad.append({
            'paper_id': p_data['paper_id'],
            'title': p_data['title'],
            'total_paper_chars': p_data['total_chars_original'],
            'executive_digest': p_data['sections']
        })

    triad_obj = {
        'rank': rank,
        'tightness_score': round(float(score), 6),
        'min_edge_cosine': round(float(t.get('min_edge', 0)), 6),
        'mean_edge_cosine': round(float(t.get('mean_edge', 0)), 6),
        'pairwise_cosine_similarities': {
            'paper1_vs_paper2': round(float(edge_weights[0]), 6),
            'paper2_vs_paper3': round(float(edge_weights[1]), 6),
            'paper1_vs_paper3': round(float(edge_weights[2]), 6),
        },
        'papers': {
            'paper_1': papers_in_triad[0],
            'paper_2': papers_in_triad[1],
            'paper_3': papers_in_triad[2],
        }
    }
    triads_out.append(triad_obj)

    # Save individual JSON
    indiv_path = json_dir / f'triad_{rank:02d}.json'
    with open(indiv_path, 'w', encoding='utf-8') as f:
        json.dump(triad_obj, f, indent=2, ensure_ascii=False)

# Save Master Executive Digest JSON
master_json_path = out_dir / 'top25_triads_executive_digest.json'
with open(master_json_path, 'w', encoding='utf-8') as f:
    json.dump({
        'n_triads': len(triads_out),
        'score_mode': hyp_data.get('score_mode'),
        'description': 'Executive Scientific Digest for top 25 triads: Abstract/Aims, Introduction/Motivation, Core Methodology, Findings/Conclusions',
        'triads': triads_out
    }, f, indent=2, ensure_ascii=False)

print(f'Successfully generated Executive Digest for {len(triads_out)} triads at: {master_json_path}')
print(f'Master size: {master_json_path.stat().st_size / 1024:.1f} KB')
