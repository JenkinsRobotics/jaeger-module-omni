"""Compare matched native speech and truth-text controls, preserving both scores."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def compare(text, speech):
    if text.get('input_mode') != 'text' or speech.get('input_mode') != 'speech':
        raise ValueError('requires text then speech reports')
    for key in ('adapter_sha256', 'case_sources_sha256', 'initial_tools',
                'system_sha256', 'case_timeout_s'):
        if not text.get(key) or text[key] != speech.get(key):
            raise ValueError('control mismatch: ' + key)
    contract = text['summary']['verified_file_outcomes'].get('contract_sha256')
    if not contract or contract != speech['summary']['verified_file_outcomes'].get('contract_sha256'):
        raise ValueError('different outcome contracts')
    def index(report):
        rows = {(r['lane'], r['id']): r for r in report['rows']}
        if len(rows) != len(report['rows']):
            raise ValueError('duplicate case identity')
        return rows
    a, b = index(text), index(speech)
    if a.keys() != b.keys():
        raise ValueError('different case populations')
    rows = []
    for key, t in a.items():
        s = b[key]
        if (t['audio_sha256'] != s['audio_sha256'] or
                t['prompt_fed_as_text'] != s['prompt_ground_truth_not_fed']):
            raise ValueError('different request or audio: ' + str(key))
        if (t.get('outcome') is None) != (s.get('outcome') is None):
            raise ValueError('different outcome coverage: ' + str(key))
        rows.append({'id': ':'.join(key),
                     'text_original_pass': t['case_pass'],
                     'speech_original_pass': s['case_pass'],
                     'text_outcome_pass': (t.get('outcome') or {}).get('pass'),
                     'speech_outcome_pass': (s.get('outcome') or {}).get('pass')})
    summary = {}
    for mode in ('text', 'speech'):
        checked = [r for r in rows if r[mode + '_outcome_pass'] is not None]
        summary[mode] = {
            'original_passed': sum(r[mode + '_original_pass'] for r in rows),
            'original_total': len(rows),
            'verified_outcome_passed': sum(r[mode + '_outcome_pass'] for r in checked),
            'verified_outcome_total': len(checked)}
    return {'summary': summary, 'rows': rows,
            'scope': 'same requests, model adapter, tool prefix and scorer; text arm receives truth text',
            'asr_vs_planning_cause_proven': False,
            'limitation': 'The gap includes recognition, punctuation ambiguity, instruction following and context effects.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('text', 'speech', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    result = compare(json.loads(args.text.read_text()), json.loads(args.speech.read_text()))
    result.update(command=[sys.executable, *sys.argv],
                  inputs_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (args.text, args.speech)},
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
