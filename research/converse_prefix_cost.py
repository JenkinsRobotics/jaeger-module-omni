"""Measure the real growing-prefix preview path for one immutable converse ear.

Each prefix re-encodes its entire waveform and generates a fresh transcript.
This is a compute probe, not physical endpoint latency or an incremental encoder.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--node', type=Path, required=True)
    p.add_argument('--audio', type=Path, required=True, help='frozen mono float32 16 kHz NPY')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from training.context_distill import refuse_gpu_contention
    refuse_gpu_contention()
    import mlx.core as mx
    from jaeger_agent_omni.node_registry import load_factory, load_package, verify_weights
    package = load_package(args.node)
    verify_weights(package)
    audio = np.asarray(np.load(args.audio), np.float32)
    if audio.ndim != 1 or len(audio) < 12*16000 or not np.isfinite(audio).all():
        raise ValueError('requires at least 12 seconds of finite mono 16 kHz audio')
    ear = load_factory(package)()
    ear.max_tokens = 128
    ear.transcribe(audio[:32000])
    features = np.random.default_rng(7).normal(size=(256, 1024)).astype(np.float32)
    full = ear.adapter(mx.array(features))
    prefix = ear.adapter(mx.array(features[:128]))
    mx.eval(full, prefix)
    drift = float(mx.max(mx.abs(full[:len(prefix)]-prefix)))
    rows = []
    for seconds in (2, 4, 6, 8, 10, 12):
        started = time.perf_counter()
        hypothesis, metrics = ear.transcribe(audio[:seconds*16000])
        rows.append({'prefix_seconds': seconds, 'hypothesis': hypothesis,
                     'request_ms': 1000*(time.perf_counter()-started), 'metrics': metrics})
        print(json.dumps(rows[-1]), flush=True)
    result = {'command': [sys.executable, *sys.argv], 'code_sha256': sha(__file__),
              'node': package.node_id, 'node_sha256': sha(args.node/'node.json'),
              'adapter_sha256': sha(args.node/package.manifest['adapter']['path']),
              'audio_sha256': sha(args.audio), 'cadence_seconds': 2, 'max_tokens': 128,
              'prefixes': rows, 'sum_request_seconds': sum(r['request_ms'] for r in rows)/1000,
              'unique_audio_seconds': 12, 'total_audio_seconds_reencoded': 42,
              'adapter_causal_prefix_max_abs_drift': drift, 'adapter_causality_probe_pass': drift < 1e-4,
              'peak_mlx_memory_bytes': mx.get_peak_memory(), 'production_changed': False,
              'scope': 'isolated growing-prefix ASR, warm model; no physical I/O or speech output',
              'limits': ['one audio fixture and one adapter prefix-integrity tensor',
                         'Whisper re-encodes prefixes; adapter causality does not make the encoder incremental']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
