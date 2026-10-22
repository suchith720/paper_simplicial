import json
import re
from pathlib import Path
from datasets import load_dataset

out_dir = Path('/home/mridul/home2/elseview/paper_simplicial/extracted_simplicial/run_english_science/hypotheses/review_top25')
json_dir = out_dir / 'abstracts_json'
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

def extract_abstract_or_lead(text: str, title: str = '') -> str:
    if not text:
        return ''

    # Search for all abstract/summary occurrences
    matches = list(re.finditer(r'(?:^|\n)\s*(?:abstract|summary)\b[:\s\-]*', text, re.IGNORECASE))
    
    for m in matches:
        start_pos = m.end()
        # Look ahead 2500 chars
        raw_snippet = text[start_pos:start_pos + 2500].strip()
        lines = [l.strip() for l in raw_snippet.split('\n') if l.strip()]
        
        # Check if this occurrence is in a table of contents
        first_lines = lines[:3]
        is_toc = any(re.search(r'\.{3,}|\bpage\b|\bchapter\b', l, re.IGNORECASE) for l in first_lines)
        if is_toc:
            continue
            
        # Collect lines until a major new section heading
        collected = []
        for line in lines:
            if re.match(r'^(?:1[\.\s]+|chapter\s+1|introduction|table\s+of\s+contents|resumen|keywords|acknowledgements|contents)\b', line, re.IGNORECASE):
                if len(collected) >= 3:
                    break
            collected.append(line)
            if sum(len(c) for c in collected) > 1800:
                break
                
        cleaned = clean_spacing('\n'.join(collected))
        if len(cleaned.split()) >= 25:
            return cleaned[:1800].strip()

    # Fallback 1: Look for Introduction heading
    m_intro = re.search(r'(?:^|\n)\s*(?:1[\.\s]+|chapter\s+1[:\s]+)?introduction\b[:\s\-]*', text, re.IGNORECASE)
    if m_intro:
        raw_snippet = text[m_intro.end():m_intro.end() + 2500].strip()
        lines = [l.strip() for l in raw_snippet.split('\n') if l.strip()]
        collected = []
        for line in lines:
            if re.match(r'^(?:2[\.\s]+|chapter\s+2|methods|background|materials)\b', line, re.IGNORECASE):
                if len(collected) >= 3:
                    break
            collected.append(line)
            if sum(len(c) for c in collected) > 1800:
                break
        cleaned = clean_spacing('\n'.join(collected))
        if len(cleaned.split()) >= 25:
            return cleaned[:1800].strip()

    # Fallback 2: Start from beginning of document, skipping single-word title/author lines
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    content_lines = [l for l in lines if len(l.split()) > 5][:15]
    cleaned = clean_spacing('\n'.join(content_lines))
    return cleaned[:1500].strip()

papers_cache = {}
for r in unique_rows:
    p = ds[int(r)]
    raw_text = p['raw_fulltext'] or ''
    extracted_lead = extract_abstract_or_lead(raw_text, str(p['title']))
    papers_cache[r] = {
        'paper_id': str(p['paper_id']),
        'title': str(p['title']),
        'extracted_abstract': extracted_lead,
        'full_text_char_length': len(raw_text),
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
            'abstract_or_lead': p_data['extracted_abstract'],
            'total_paper_chars': p_data['full_text_char_length'],
            'dataset_row': r,
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

    # Individual human-readable JSON per triad
    indiv_path = json_dir / f'triad_{rank:02d}.json'
    with open(indiv_path, 'w', encoding='utf-8') as f:
        json.dump(triad_obj, f, indent=2, ensure_ascii=False)

# Master readable JSON
master_json_path = out_dir / 'top25_triads_readable_abstracts.json'
with open(master_json_path, 'w', encoding='utf-8') as f:
    json.dump({
        'n_triads': len(triads_out),
        'score_mode': hyp_data.get('score_mode'),
        'description': 'Human-readable top 25 triads with paper titles, extracted abstracts/leads, and pairwise similarity scores',
        'triads': triads_out
    }, f, indent=2, ensure_ascii=False)

print(f"Generated master JSON: {master_json_path} (size: {master_json_path.stat().st_size / 1024:.1f} KB)")
