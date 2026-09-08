"""Reconstruct exact WER edits and expose failure slices without relabeling audio."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.evaluate import normalize


def alignment(reference, hypothesis):
    r, h = normalize(reference), normalize(hypothesis)
    costs = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1): costs[i][0] = i
    for j in range(len(h) + 1): costs[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            costs[i][j] = min(costs[i-1][j-1] + (r[i-1] != h[j-1]),
                              costs[i-1][j] + 1, costs[i][j-1] + 1)
    edits = []
    i, j = len(r), len(h)
    while i or j:
        if i and j and costs[i][j] == costs[i-1][j-1] + (r[i-1] != h[j-1]):
            if r[i-1] != h[j-1]: edits.append(('substitution', r[i-1], h[j-1]))
            i -= 1; j -= 1
        elif i and costs[i][j] == costs[i-1][j] + 1:
            edits.append(('deletion', r[i-1], '')); i -= 1
        else:
            edits.append(('insertion', '', h[j-1])); j -= 1
    return list(reversed(edits))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source = json.loads(args.input.read_text())
    rows = []
    for row in source['rows']:
        if row['task'] != 'asr': continue
        edits = alignment(row['reference'], row['hypothesis'])
        if len(edits) != row['edits']:
            raise ValueError('alignment disagrees with fixed scorer: ' + row['id'])
        rows.append({**{k: row[k] for k in ('id','corpus','seconds','reference','hypothesis','reference_words')},
                     'edits': edits,
                     'decode_cap_hit': row.get('metrics', {}).get('generated_tokens', -1) >= source['max_tokens']['asr']})
    slices = {}
    for corpus in sorted({r['corpus'] for r in rows}):
        group = [r for r in rows if r['corpus'] == corpus]
        counts = Counter(e[0] for r in group for e in r['edits'])
        common = Counter(tuple(e) for r in group for e in r['edits'])
        slices[corpus] = {'rows':len(group), 'words':sum(r['reference_words'] for r in group),
                          'edit_types':dict(counts), 'decode_cap_hits':sum(r['decode_cap_hit'] for r in group),
                          'frequent_edits':[{'edit':e, 'count':n} for e,n in common.most_common(20)]}
    report = {'command':[sys.executable,*sys.argv],
              'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'slices':slices, 'rows':rows,
              'annotation_limits':'corpus provenance only; noise, overlap and proper-name labels require independent audio annotation',
              'promotion_decision':'none; diagnostic, not a new held-out population'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(slices, indent=2))


if __name__ == '__main__': main()
