"""Offline, paired diagnostic for spoken integers rejected by corpus substrings.

Preserves the original scored JSON. This adds only exact, complete English
cardinal phrases for integer expectations; it does not change WER or the
standing corpus gate. Apply the same function to every compared report.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import benchmark_agentic_corpus as corpus

ONES = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
NUMBER_WORDS = set(ONES + TENS[2:] + ['hundred', 'thousand', 'million', 'billion', 'and', 'point', 'minus', 'negative', '-', '−'])


def integer_words(n):
    if n < 20:
        return [ONES[n]]
    if n < 100:
        return [TENS[n // 10]] + (integer_words(n % 10) if n % 10 else [])
    for size, name in [(10**9, 'billion'), (10**6, 'million'), (1000, 'thousand'), (100, 'hundred')]:
        if n >= size:
            return integer_words(n // size) + [name] + (integer_words(n % size) if n % size else [])
    raise AssertionError(n)


def spoken_integer_match(answer, expected):
    if not re.fullmatch(r'\d{1,12}', expected):
        return False
    answer = re.sub(r'(?<=[a-zA-Z])-(?=[a-zA-Z])', ' ', answer)
    wanted = integer_words(int(expected))
    run = []
    for word in re.findall(r'[a-z]+|\d+|[-−]', answer.lower()) + ['<end>']:
        if word in NUMBER_WORDS:
            run.append(word)
        else:
            if [w for w in run if w != 'and'] == wanted:
                return True
            run = []
    return False


def numeric_contains(answer, needles, *, all_values):
    if not needles:
        return True
    found = [corpus_contains(answer, [n], all_values=False) or
             spoken_integer_match(answer, n) for n in needles]
    return all(found) if all_values else any(found)


corpus_contains = corpus._contains


def audit(report):
    cases = {(lane, case.id): case for lane, case in corpus._selected('both', [], [], 0)}
    added = []
    rows = []
    with patch.object(corpus, '_contains', numeric_contains):
        for row in report['rows']:
            score = corpus._score(cases[row['lane'], row['id']], row['tools_called'],
                                  row['skills_viewed'], row['answer'], row['error'])
            rows.append({**row, **score})
            if score['case_pass'] and not row['case_pass']:
                added.append({'lane': row['lane'], 'id': row['id'], 'answer': row['answer']})
    return {'original_summary': report['summary'], 'diagnostic_summary': corpus.summarize(rows),
            'additional_integer_equivalence_passes': added}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reports', type=Path, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    reports = [json.loads(path.read_text()) for path in args.reports]
    populations = [[(r['lane'], r['id'], r['audio_sha256']) for r in d['rows']] for d in reports]
    if not all(x == populations[0] for x in populations):
        raise ValueError('Comparison requires identical ordered cases and audio')
    output = {'kind': 'paired_spoken_integer_diagnostic', 'original_gate_unchanged': True,
              'same_ordered_audio_population': True, 'command': [sys.executable, *sys.argv],
              'scorer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'reports': {str(path): audit(d) for path, d in zip(args.reports, reports)}}
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k: v['diagnostic_summary']['passed'] for k,v in output['reports'].items()}))


if __name__ == '__main__':
    main()
