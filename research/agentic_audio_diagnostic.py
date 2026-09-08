"""Freeze an ASR diagnostic from the exact audio used by an action report.

This does not feed a transcript back into the agent. Run the resulting manifest
with converse_ear_wer.py to distinguish transcription errors from action errors.
The separate ASR prompt cannot certify what the agent's hidden state contained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from jaeger_agent_omni.app import read_wav
    import soundfile as sf
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=Path.home()/'models/jaeger-training-data/agentic-asr-v1')
    p.add_argument('--manifest', type=Path, required=True)
    args = p.parse_args()
    report = json.loads(args.report.read_text())
    if report.get('input_mode') != 'speech':
        raise ValueError('requires a native speech report')
    args.cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in report['rows']:
        source = Path(row['audio'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != row['audio_sha256']:
            raise ValueError('fixture changed: ' + str(source))
        key = row['lane'] + '-' + row['id']
        target = args.cache / (key + '-' + row['audio_sha256'][:16] + '.wav')
        # Float WAV preserves the runtime's resampled float32 samples exactly.
        sf.write(target, read_wav(source), 16000, subtype='FLOAT')
        rows.append({'id': key, 'audio': str(target),
                     'text': row['prompt_ground_truth_not_fed'],
                     'audio_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                     'original_audio_sha256': row['audio_sha256']})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    audit = {'command': [sys.executable, *sys.argv], 'rows': len(rows),
             'report_sha256': hashlib.sha256(args.report.read_bytes()).hexdigest(),
             'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
             'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'runtime_audio_reader_sha256': hashlib.sha256((ROOT/'jaeger_agent_omni/app.py').read_bytes()).hexdigest(),
             'scope': 'same resampled waveforms, separate ASR instruction; no transcript bridge'}
    args.manifest.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
