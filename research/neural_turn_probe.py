"""CPU gain/noise diagnostic for the independent neural turn-state candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from jaeger_agent_omni.node_registry import load_factory, load_package
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio', type=Path, default=ROOT/'jaeger_agent_omni/tests/simcache/hey_computer_what_is_two_plus_two_.npy')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    audio = np.load(args.audio).astype(np.float32)
    factories = {name: load_factory(load_package(ROOT/'jaeger_agent_omni/nodes'/name))
                 for name in ('para-turn-state-v0', 'para-turn-state-neural-v1')}
    rows = []
    for kind in ('speech', 'noise'):
        raw = audio if kind == 'speech' else np.random.default_rng(7).normal(size=len(audio)).astype(np.float32)
        for rms in (.002, .006, .02, .05):
            signal = raw * (rms/float(np.sqrt(np.mean(raw**2))))
            signal = np.concatenate([signal, np.zeros(16000, np.float32)])
            signal = np.pad(signal, (0, (-len(signal)) % 480))
            for name, factory in factories.items():
                node = factory()
                events, costs = [], []
                for offset in range(0, len(signal), 480):
                    t = time.perf_counter()
                    state = node.process(signal[offset:offset+480], offset/16000)
                    costs.append(1000*(time.perf_counter()-t))
                    if state['turn_ended']:
                        events.append((offset+480)/16000)
                rows.append({'kind': kind, 'rms': rms, 'node': name, 'endpoints_seconds': events,
                             'expected_one_or_more': kind == 'speech',
                             'detected_as_expected': bool(events) == (kind == 'speech'),
                             'process_p95_ms': float(np.percentile(costs, 95))})
    sources = [Path(__file__), args.audio, ROOT/'jaeger_agent_omni/core/runtime/endpoint.py']
    sources += [ROOT/'jaeger_agent_omni/nodes'/name/file for name in factories for file in ('node.json','runtime.py')]
    report = {'command': [sys.executable, *sys.argv],
              'sources_sha256': {str(v): hashlib.sha256(v.read_bytes()).hexdigest() for v in sources},
              'rows': rows, 'scope': 'one cached synthetic phrase at four gains plus seeded noise; no AEC, overlap or physical claims',
              'production_changed': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    main()
