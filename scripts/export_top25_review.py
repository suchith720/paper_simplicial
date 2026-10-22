import json
import os
from pathlib import Path
from datasets import load_dataset

out_dir = Path('/home/mridul/home2/elseview/paper_simplicial/extracted_simplicial/run_english_science/hypotheses/review_top25')
md_dir = out_dir / 'triads_md'
out_dir.mkdir(parents=True, exist_ok=True)
md_dir.mkdir(parents=True, exist_ok=True)

hyp_json_path = '/home/mridul/home2/elseview/paper_simplicial/extracted_simplicial/run_english_science/hypotheses/clique_english_tight_top1000.json'
with open(hyp_json_path) as f:
    hyp_data = json.load(f)

triangles = hyp_data['triangles'][:25]
unique_rows = sorted(list({r for t in triangles for r in t['row_idx']}))

print(f'Loading dataset for {len(unique_rows)} unique papers...')
ds = load_dataset('scientifi-papers/scientific-papers', 'corex')['train']

papers_cache = {}
for r in unique_rows:
    p = ds[int(r)]
    raw_text = p['raw_fulltext'] or ''
    papers_cache[r] = {
        'paper_id': str(p['paper_id']),
        'title': str(p['title']),
        'char_length': len(raw_text),
        'full_text': raw_text
    }

print('Assembling top 25 fulltext review dataset...')
triads_out = []
readme_lines = [
    '# Top 25 Paper Triads Review Package (Full Text)',
    '',
    'This directory contains the unabridged full text of all papers in the top 25 ranked triads from the `scientific-papers/corex` dataset.',
    '',
    '| Rank | Score | Paper 1 Title | Paper 2 Title | Paper 3 Title | Detailed File |',
    '| :---: | :---: | :--- | :--- | :--- | :--- |'
]

for idx, t in enumerate(triangles, 1):
    rank = t['rank']
    score = t['score']
    edge_weights = t['edge_weights'] # [w12, w23, w13]
    titles = t['titles']
    paper_ids = t['paper_ids']
    row_idxs = t['row_idx']

    papers_in_triad = []
    for r, pid, tit in zip(row_idxs, paper_ids, titles):
        p_data = papers_cache[r]
        papers_in_triad.append({
            'row_idx': r,
            'paper_id': p_data['paper_id'],
            'title': p_data['title'],
            'char_length': p_data['char_length'],
            'full_text': p_data['full_text']
        })

    triad_obj = {
        'rank': rank,
        'score': score,
        'min_edge': t.get('min_edge'),
        'mean_edge': t.get('mean_edge'),
        'edge_weights': {
            'paper1_paper2': edge_weights[0],
            'paper2_paper3': edge_weights[1],
            'paper1_paper3': edge_weights[2],
        },
        'papers': papers_in_triad
    }
    triads_out.append(triad_obj)

    # Write individual triad JSON
    triad_json_file = out_dir / f'triad_{rank:02d}.json'
    with open(triad_json_file, 'w', encoding='utf-8') as f:
        json.dump(triad_obj, f, indent=2, ensure_ascii=False)

    # Write individual triad Markdown
    md_file_rel = f'triads_md/triad_{rank:02d}.md'
    md_file = out_dir / md_file_rel
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# Triad Rank #{rank} (Tightness Score: {score:.6f})\n\n")
        f.write(f"- **Min Edge Cosine**: {t.get('min_edge', 0):.6f}\n")
        f.write(f"- **Mean Edge Cosine**: {t.get('mean_edge', 0):.6f}\n")
        f.write(f"- **Pairwise Similarities**:\n")
        f.write(f"  - Paper 1 <-> Paper 2: `{edge_weights[0]:.6f}`\n")
        f.write(f"  - Paper 2 <-> Paper 3: `{edge_weights[1]:.6f}`\n")
        f.write(f"  - Paper 1 <-> Paper 3: `{edge_weights[2]:.6f}`\n\n")
        f.write("---\n\n")

        for p_idx, p in enumerate(papers_in_triad, 1):
            f.write(f"## Paper {p_idx}: {p['title']}\n\n")
            f.write(f"- **Paper ID**: `{p['paper_id']}` (dataset row `{p['row_idx']}`)\n")
            f.write(f"- **Character Length**: {p['char_length']:,} characters\n\n")
            f.write("### Full Text\n\n")
            f.write("```text\n")
            f.write(p['full_text'].strip())
            f.write("\n```\n\n---\n\n")

    readme_lines.append(
        f"| **#{rank}** | `{score:.4f}` | {titles[0][:60]}... | {titles[1][:60]}... | {titles[2][:60]}... | [{md_file_rel}]({md_file_rel}) |"
    )

# Write master combined JSON
combined_json = out_dir / 'top25_triads_fulltext.json'
with open(combined_json, 'w', encoding='utf-8') as f:
    json.dump({
        'n_triads': len(triads_out),
        'scoring_mode': hyp_data.get('score_mode'),
        'description': 'Full text attached for top 25 paper triads from corex scientific papers',
        'triads': triads_out
    }, f, indent=2, ensure_ascii=False)

# Write master README
with open(out_dir / 'README.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(readme_lines))

print(f"Successfully created review package at: {out_dir}")
print(f"Master JSON size: {combined_json.stat().st_size / (1024*1024):.2f} MB")
