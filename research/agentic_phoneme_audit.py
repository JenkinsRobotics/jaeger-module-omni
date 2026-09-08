"""Audit synthetic action prompts before attributing every error to the ear.

No waveform or benchmark result is changed. Empty phonemes expose information
lost before synthesis; nonempty phonemes do not prove acoustic intelligibility.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from kokoro import KPipeline
    torch.set_num_threads(2)
    pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', model=False)
    source = json.loads(args.report.read_text())
    if source.get('input_mode') != 'speech':
        raise ValueError('requires a speech report')
    rows = []
    for row in source['rows']:
        request = row['prompt_ground_truth_not_fed']
        phonemes, tokens = pipeline.g2p(request)
        rows.append({'id': row['lane'] + ':' + row['id'], 'text': request,
                     'phonemes': phonemes,
                     'tokens': [{'text': t.text, 'phonemes': t.phonemes} for t in tokens],
                     'empty_alphanumeric_or_underscore': [t.text for t in tokens
                         if not t.phonemes and any(c.isalnum() or c == '_' for c in t.text)]})
    report = {'command': [sys.executable, *sys.argv],
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'source_report': str(args.report),
              'source_sha256': hashlib.sha256(args.report.read_bytes()).hexdigest(),
              'libraries': {name: version(name) for name in ('kokoro', 'misaki')},
              'scope': 'current phonemizer audit; no waveform listening claim; frozen results unchanged',
              'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({r['id']: r['empty_alphanumeric_or_underscore'] for r in rows
                      if r['empty_alphanumeric_or_underscore']}))


if __name__ == '__main__':
    main()
