"""Experimental image/text/audio turns in one unchanged MLX language-model cache.

prepare runs the separate frozen vision NN and Kokoro fixtures on CPU. run uses
the canonical medium ear and existing brain; no image caption/transcript bridge.
Reported request latency excludes the separately measured vision preprocessing.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = Path.home() / 'models/jaeger-training-data/joint-vision-v1'
CASES = [('green', 'triangle', 'cedar'), ('orange', 'square', 'harbor'),
         ('purple', 'circle', 'maple'), ('red', 'rectangle', 'river')]
VOICE = {
    'color': 'What color was the shape in the most recent image?',
    'label': 'What label did I give the most recent shape?',
    'joint': 'What shape and color are shown in this image?',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def multimodal_turn(session, meta, *, features=None, boundaries=None, text=None,
                    audio=None, max_tokens=96):
    """Only input modalities are changed; the session/model/cache are reused."""
    if features is not None:
        features = np.asarray(features, np.float32)
        if features.ndim != 2 or features.shape[1] != 2560 or not len(features) or not np.isfinite(features).all():
            raise ValueError('expected finite image tokens with width 2560')
        if boundaries is None or len(boundaries) != 2:
            raise ValueError('expected image start/end token IDs')
    session.open_turn()
    session._ensure_turn_prefix()
    if features is not None:
        session._prefill([boundaries[0]])
        # mlx_lm applies sqrt(hidden_size) inside the language model; the
        # native VLM inserts projected visual embeddings after that scaling.
        session._prefill([0] * len(features), session.mx.array(features) / math.sqrt(2560))
        session._prefill([boundaries[1]])
    if text:
        session._prefill(session.tokenizer.encode(text, add_special_tokens=False))
    if audio is not None:
        for first in range(0, len(audio), 1600):
            session.feed_audio(audio[first:first+1600])
        session.end_of_turn()
    else:
        _, suffix = session._prompt_parts()
        session._prefill(suffix[:-1])
        session._response_seed = int(suffix[-1])
        session._turn_closed = True
    return session.respond(max_tokens=max_tokens, eos_ids=meta['eos'])


def prepare(args):
    import mlx.core as mx
    import torch
    from PIL import Image, ImageDraw
    from jaeger_agent_omni.node_registry import load_factory, load_package
    from jaeger_agent_omni.voice import speak_kokoro
    mx.set_default_device(mx.cpu)
    torch.set_num_threads(2)
    node = ROOT / 'jaeger_agent_omni/nodes/gemma4-vision-mlx-v1'
    vision = load_factory(load_package(node))()
    args.cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for color, shape, label in CASES:
        image = Image.new('RGB', (512, 512), 'white')
        draw = ImageDraw.Draw(image)
        if shape == 'triangle':
            draw.polygon([(256, 65), (65, 440), (447, 440)], fill=color)
        elif shape == 'circle':
            draw.ellipse((80, 80, 432, 432), fill=color)
        else:
            draw.rectangle((80, 80 if shape == 'square' else 170, 432,
                            432 if shape == 'square' else 342), fill=color)
        path = args.cache / (shape + '.png')
        image.save(path)
        start = time.perf_counter()
        features = vision.encode(image)
        seconds = time.perf_counter() - start
        feature_path = args.cache / (shape + '.npy')
        np.save(feature_path, features)
        rows.append({'shape': shape, 'color': color, 'label': label,
                     'image': str(path), 'image_sha256': sha(path),
                     'features': str(feature_path), 'features_sha256': sha(feature_path),
                     'tokens': len(features), 'vision_cpu_seconds': seconds})
        print(json.dumps(rows[-1]), flush=True)
    voices = {}
    for key, words in VOICE.items():
        path = args.cache / (key + '.wav')
        if not path.exists():
            speak_kokoro(words, path, voice='am_adam', speed=1.0)
        voices[key] = {'audio': str(path), 'sha256': sha(path), 'words_not_fed_as_text': words}
    result = {'command': [sys.executable, *sys.argv], 'vision_node': node.name,
              'node_sha256': sha(node/'node.json'), 'vision_runtime_sha256': sha(node/'runtime.py'),
              'code_sha256': sha(__file__), 'device': 'cpu', 'language_model_loaded': False,
              'boundaries': [vision.image_start_id, vision.image_end_id],
              'rows': rows, 'voice': voices}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2)+'\n')


def run(args):
    from training.context_distill import refuse_gpu_contention
    refuse_gpu_contention()
    from jaeger_agent_omni.app import read_wav
    from jaeger_agent_omni.omni import MODEL, SYSTEM, build
    manifest = json.loads(args.manifest.read_text())
    for row in manifest['rows']:
        for key in ('image', 'features'):
            if sha(row[key]) != row[key+'_sha256']:
                raise ValueError('fixture changed: ' + row[key])
    for row in manifest['voice'].values():
        if sha(row['audio']) != row['sha256']:
            raise ValueError('voice fixture changed')
    session, meta = build(MODEL, 'jaegeromni-41m-medium', system=SYSTEM)
    session.warmup()
    brain_id = id(session.model)
    session.reset()
    source_snapshot = {str(p.relative_to(ROOT)): sha(p) for p in (
        Path(__file__), ROOT/'jaeger_agent_omni/core/runtime/stock_whisper.py')}
    rows = []
    for case in manifest['rows']:
        features = np.load(case['features'])
        steps = [
            ('image_text', 'Name the shape and its color in this image.', None, True, [case['color'], case['shape']]),
            ('spoken_color', None, 'color', False, [case['color']]),
            ('typed_label', f"Use the label {case['label']} for that shape from now on. What label will you use?", None, False, [case['label']]),
            ('spoken_label', None, 'label', False, [case['label']]),
            ('image_audio', None, 'joint', True, [case['color'], case['shape']]),
        ]
        for kind, text, voice, image, expected in steps:
            before = session.n_prefilled
            start = time.perf_counter()
            error = None
            try:
                answer = multimodal_turn(session, meta,
                    features=features if image else None, boundaries=manifest['boundaries'],
                    text=text, audio=read_wav(manifest['voice'][voice]['audio']) if voice else None)
            except Exception as exc:
                answer = ''
                error = f'{type(exc).__name__}: {exc}'
            rows.append({'shape': case['shape'], 'kind': kind, 'text_fed': text,
                         'voice_fixture': voice, 'image_fed': image, 'expected': expected,
                         'answer': answer, 'error': error,
                         'pass': error is None and all(w in answer.lower() for w in expected),
                         'request_ms': 1000*(time.perf_counter()-start),
                         'cache_before': before, 'cache_after': session.n_prefilled,
                         'same_brain_instance': id(session.model) == brain_id})
            print(json.dumps(rows[-1]), flush=True)
            if error:
                break
        if rows[-1]['error']:
            break
    result = {'command': [sys.executable, *sys.argv], 'source_snapshot': source_snapshot,
              'manifest_sha256': sha(args.manifest), 'adapter_sha256': meta.get('adapter_sha256'),
              'brain': meta['model'], 'brain_instances': 1,
              'passed': sum(r['pass'] for r in rows), 'total': len(rows), 'expected_total': 20,
              'all_pass': len(rows) == 20 and all(r['pass'] and r['same_brain_instance'] for r in rows),
              'rows': rows, 'production_changed': False,
              'scope': 'synthetic shapes; frozen CPU-cached vision embeddings plus raw speech and text in one MLX cache',
              'latency_scope': 'excludes separately measured vision encoding and model loading',
              'attention_scope': 'causal language attention; no image-only bidirectional attention overlay',
              'caption_or_transcript_bridge': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    return 0 if result['all_pass'] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run'])
    p.add_argument('--cache', type=Path, default=CACHE)
    p.add_argument('--manifest', type=Path, default=ROOT/'research/candidate_evals/joint-vision-fixtures-20260907.json')
    p.add_argument('--output', type=Path, default=ROOT/'research/candidate_evals/joint-vision-probe-20260907.json')
    args = p.parse_args()
    return prepare(args) if args.action == 'prepare' else run(args)


if __name__ == '__main__':
    raise SystemExit(main())
