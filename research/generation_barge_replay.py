"""CPU-only replay of the pre-output barge speech gate; no overlap-F1 claim."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jaeger_agent_omni.core.runtime.neural_vad import load_silero, SpeechPresence


def run(audio):
    engine, package = load_silero()
    gate = SpeechPresence(engine.probability,
                          frame_samples=int(package.manifest['frame_samples']))
    rows = []
    for start in range(0, len(audio), 1600):
        block = audio[start:start+1600]
        rms = float(np.sqrt(np.mean(block**2)))
        rows.append({'start_s': start/16000, 'rms': rms,
                     'legacy_energy': rms > 0.012*1.1,
                     'neural_speech': gate.update(block)})
    return rows, gate.summary()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--actor', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    captured, rate = sf.read(args.capture, dtype='float32')
    assert rate == 16000 and captured.ndim == 1
    actor, rate = sf.read(args.actor, dtype='float32')
    actor = resample_poly(actor, 16000, rate).astype(np.float32)
    noise = np.random.default_rng(11).normal(0, .07, 5*16000).astype(np.float32)
    reports = {}
    for name, audio in [('captured_pre_actor_5s', captured[:5*16000]),
                        ('seed11_broadband_noise', noise), ('clean_actor', actor)]:
        rows, timings = run(audio)
        reports[name] = {'blocks': len(rows),
                         'legacy_energy_active_blocks': sum(r['legacy_energy'] for r in rows),
                         'neural_speech_active_blocks': sum(r['neural_speech'] for r in rows),
                         'first_neural_active_block_end_s': next((r['start_s']+.1 for r in rows if r['neural_speech']), None),
                         'timings': timings, 'rows': rows}
    passed = (reports['seed11_broadband_noise']['neural_speech_active_blocks'] == 0 and
              reports['clean_actor']['neural_speech_active_blocks'] > 0 and
              reports['captured_pre_actor_5s']['neural_speech_active_blocks'] <
              reports['captured_pre_actor_5s']['legacy_energy_active_blocks'])
    result = {'kind': 'generation_barge_cpu_replay', 'passed': passed,
              'command': [sys.executable, *sys.argv], 'reports': reports,
              'source_sha256': {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                                for f in [Path(__file__), ROOT/'jaeger_agent_omni/core/runtime/neural_vad.py']},
              'audio_sha256': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in [args.capture, args.actor]},
              'limitations': ['Pre-actor recording can contain uncontrolled room sounds.',
                              'The first detection is measured from clip start, not annotated speech onset.',
                              'Speech detection is not speaker overlap detection or an AEC-tail fix.']}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: {m:v for m,v in r.items() if m != 'rows'} for k,r in reports.items()}, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
