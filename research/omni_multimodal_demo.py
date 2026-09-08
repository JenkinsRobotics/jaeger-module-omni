"""Run local image/audio/text turns in one frozen Gemma, optionally saving speech.

Explicit experimental entrypoint; no production profile changes. A turns JSON
contains a list of objects with image, audio and/or text fields. Image and audio
paths are relative to that JSON. No caption/transcript is passed between models.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.joint_vision_probe import multimodal_turn, sha


def load_turns(path):
    turns = json.loads(path.read_text())
    if not isinstance(turns, list) or not turns:
        raise ValueError('turns must be a nonempty JSON list')
    result = []
    for turn in turns:
        if not isinstance(turn, dict) or set(turn) - {'image', 'audio', 'text'}:
            raise ValueError('each turn accepts only image, audio and text')
        if not any(turn.get(k) for k in ('image', 'audio', 'text')):
            raise ValueError('empty turn')
        if any(not isinstance(value, str) for value in turn.values()):
            raise ValueError('turn values must be strings')
        row = dict(turn)
        for key in ('image', 'audio'):
            if row.get(key):
                source = (path.parent / row[key]).resolve()
                if not source.is_file():
                    raise ValueError(f'missing {key}: {source}')
                row[key] = str(source)
        result.append(row)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--turns', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--speak', action='store_true', help='save Kokoro reply WAVs next to report; no device playback')
    p.add_argument('--max-tokens', type=int, default=96)
    args = p.parse_args()
    if args.max_tokens <= 0:
        p.error('--max-tokens must be positive')
    turns = load_turns(args.turns)
    from training.context_distill import refuse_gpu_contention
    refuse_gpu_contention()
    import mlx.core as mx
    from PIL import Image
    from jaeger_agent_omni.app import read_wav
    from jaeger_agent_omni.node_registry import load_factory, load_package
    from jaeger_agent_omni.omni import MODEL, SYSTEM, build
    start = time.perf_counter()
    session, meta = build(MODEL, 'jaegeromni-41m-medium', system=SYSTEM)
    session.warmup()
    vision = None
    if any(t.get('image') for t in turns):
        vision = load_factory(load_package(ROOT/'jaeger_agent_omni/nodes/gemma4-vision-mlx-v1'))()
    load_seconds = time.perf_counter()-start
    brain_id = id(session.model)
    identity = {str(path.relative_to(ROOT)): sha(path) for path in (
        ROOT/'jaeger_agent_omni/nodes/jaegeromni-41m-medium/node.json',
        ROOT/'jaeger_agent_omni/nodes/gemma4-vision-mlx-v1/node.json',
        ROOT/'jaeger_agent_omni/nodes/gemma4-vision-mlx-v1/runtime.py',
        ROOT/'jaeger_agent_omni/core/runtime/stock_whisper.py')}
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, turn in enumerate(turns):
        start = time.perf_counter()
        features, vision_ms, error, answer = None, 0.0, None, ''
        try:
            if turn.get('image'):
                encode_start = time.perf_counter()
                with Image.open(turn['image']) as picture:
                    features = vision.encode(picture.convert('RGB'))
                vision_ms = 1000*(time.perf_counter()-encode_start)
            answer = multimodal_turn(session, meta, features=features,
                boundaries=[vision.image_start_id, vision.image_end_id] if features is not None else None,
                text=turn.get('text'), audio=read_wav(turn['audio']) if turn.get('audio') else None,
                max_tokens=args.max_tokens)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        record = {'input': turn, 'input_sha256': {k: sha(turn[k]) for k in ('image', 'audio') if turn.get(k)},
                  'answer': answer, 'error': error, 'vision_ms': vision_ms,
                  'request_ms_including_vision': 1000*(time.perf_counter()-start),
                  'same_brain_instance': id(session.model) == brain_id,
                  'cache_tokens': session.n_prefilled}
        if args.speak and answer and not error:
            from jaeger_agent_omni.voice import speak_kokoro
            target = args.output.with_suffix('.audio') / f'reply-{index:03d}.wav'
            start = time.perf_counter()
            try:
                speak_kokoro(answer, target, voice='af_heart', speed=1.0)
                record['speech'] = {'path': str(target), 'sha256': sha(target),
                                    'synthesis_ms': 1000*(time.perf_counter()-start)}
            except Exception as exc:
                record['error'] = f'synthesis: {type(exc).__name__}: {exc}'
        records.append(record)
        report = {'command': [sys.executable, *sys.argv], 'code_sha256': sha(__file__),
                  'helper_sha256': sha(ROOT/'research/joint_vision_probe.py'),
                  'turns_sha256': sha(args.turns), 'brain': meta['model'],
                  'runtime_identity_sha256': identity,
                  'front_end': 'jaegeromni-41m-medium', 'vision_node': 'gemma4-vision-mlx-v1' if vision else None,
                  'brain_instances': 1, 'caption_or_transcript_bridge': False,
                  'load_and_warmup_seconds': load_seconds,
                  'peak_mlx_memory_bytes': mx.get_peak_memory(), 'rows': records,
                  'production_profile_changed': False,
                  'scope': 'experimental causal-attention multimodal session; saved speech is not physical delivery'}
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(record), flush=True)
        if record['error']:
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
