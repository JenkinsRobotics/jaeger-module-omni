"""Paired PCM/scorer benchmark, one model per process, with frozen fixtures.

prepare is CPU-only. run requires the GPU guard. Comparison covers measured
rows only; ASR/answer probes do not establish full-system superiority.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'research'))
from training.evaluate import normalize, wer
from grading import graded

ASR_SYSTEM = 'Repeat the spoken words exactly. Output only what was said, with no commentary or explanation.'
CACHE = Path.home()/'models/jaeger-training-data/paired-omni-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def prepare(args):
    import soundfile as sf
    data = Path.home()/'models/jaeger-training-data'
    paths = {'devclean': data/'stock-whisper-medium-eval-v1/devclean/manifest.jsonl',
             'ami': data/'stock-whisper-medium-expanded-v1/ami5_train_validation/manifest.jsonl'}
    cases, inputs = [], {}
    for corpus, path in paths.items():
        inputs[corpus] = sha(path)
        rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        rows = [r for r in rows if str(r.get('text') or '').strip()][:64]
        for row in rows:
            cases.append({'id': corpus+':'+row['id'], 'task': 'asr', 'corpus': corpus,
                          'source_audio': row['audio'], 'reference': row['text'],
                          'source_id': row['id'], 'speaker': row.get('speaker')})
    path = ROOT/'research/cases12/cases.json'
    inputs['cases12'] = sha(path)
    for row in json.loads(path.read_text()):
        cases.append({'id': 'cases12:'+row['key'], 'task': 'answer', 'corpus': 'cases12',
                      'source_audio': str(path.parent/(row['key']+'.wav')),
                      'reference': row['transcript'], 'answers': row['answers']})
    args.cache.mkdir(parents=True, exist_ok=True)
    for row in cases:
        audio, rate = sf.read(row['source_audio'], dtype='float32')
        if rate != 16000 or audio.ndim != 1:
            raise ValueError(f"expected mono 16 kHz: {row['source_audio']}")
        target = args.cache/(hashlib.sha256(row['id'].encode()).hexdigest()[:20]+'.wav')
        sf.write(target, audio, rate, subtype='PCM_16')
        row.update(audio=str(target), audio_sha256=sha(target), seconds=len(audio)/rate,
                   source_audio_sha256=sha(row['source_audio']))
    report = {'schema': 1, 'sources': inputs, 'cases': cases,
              'population': 'existing development gates; AMI training-adjacent; no fresh lockbox',
              'command': [sys.executable, *sys.argv]}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Frozen {len(cases)} fixtures: {args.manifest}', flush=True)


class MiniCPM:
    def __init__(self, work, template='system'):
        from minicpm_stages import load_shim, LLM_GGUF, REF_AUDIO, SHIM
        self.lib, self.reference = load_shim(), REF_AUDIO
        self.work = work
        self.template = template
        self.tts = template == 'user-tts'
        self.reset_safe = True
        work.mkdir(parents=True, exist_ok=True)
        source = ROOT/'research/minicpm_benchmark_bridge.cpp'
        engine = Path.home()/'GITHUB/llama.cpp-omni'
        engine_header = engine/'tools/omni/omni.h'
        engine_lib = engine/'build/bin/libomni.dylib'
        bridge_key = hashlib.sha256((sha(source)+sha(engine_header)+sha(engine_lib)).encode()).hexdigest()[:12]
        bridge = ROOT/'artifacts/paired-omni'/('bridge-'+bridge_key+'.dylib')
        if not bridge.exists():
            subprocess.run(['clang++','-std=c++17','-O2','-shared','-fPIC',
                            *['-I'+str(engine/p) for p in ('tools/omni','common','include','ggml/include','vendor','src')],
                            str(source),'-o',str(bridge),'-L'+str(engine/'build/bin'),
                            '-lomni','-lllama','-lllama-common','-lggml','-lggml-base',
                            '-Wl,-rpath,'+str(engine/'build/bin')],check=True)
        self.bridge = ctypes.CDLL(str(bridge))
        v, c, i = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int
        self.bridge.paired_c_audio_instruction.argtypes = [v,c,i]
        self.bridge.paired_c_audio_instruction.restype = None
        self.bridge.paired_c_audio_prompt.argtypes = [v]
        self.bridge.paired_c_audio_prompt.restype = c
        for name, sig in [('omni_c_reset_context', [v]),
                          ('omni_c_append_system_prompt', [v,c]),
                          ('omni_c_configure_decode', [v,i,ctypes.c_float])]:
            f = getattr(self.lib, name); f.argtypes = sig; f.restype = None
        self.lib.omni_c_drain_audio.argtypes = [v,i,i]
        self.lib.omni_c_drain_audio.restype = i
        self.ctx = self.lib.omni_c_init(str(LLM_GGUF).encode(), 1, int(self.tts), 0, 8192, 4, str(work).encode())
        if not self.ctx:
            raise RuntimeError('MiniCPM initialization failed')
        self.identity = {'model': str(LLM_GGUF), 'model_sha256': sha(LLM_GGUF),
                         'shim_sha256': sha(SHIM), 'voice_reference_used': False,
                         'bridge_source_sha256': sha(source), 'bridge_sha256': sha(bridge),
                         'engine_header_sha256': sha(engine_header), 'engine_library_sha256': sha(engine_lib),
                         'tts': self.tts, 'template': template,
                         'runtime': 'local llama.cpp-omni ctypes shim'}

    def infer(self, row, system, max_tokens):
        if not self.reset_safe:
            raise RuntimeError('previous TTS job did not drain; refusing context reuse')
        self.lib.omni_c_reset_context(self.ctx)
        self.lib.omni_c_set_language(self.ctx, b'en')
        self.bridge.paired_c_audio_instruction(self.ctx, system.encode(), int(self.template != 'system'))
        if system not in self.bridge.paired_c_audio_prompt(self.ctx).decode():
            raise RuntimeError('task instruction missing from actual audio prompt')
        self.lib.omni_c_configure_decode(self.ctx, max_tokens, 0.0)
        if not self.lib.omni_c_stream_prefill(self.ctx, None, 0):
            raise RuntimeError('MiniCPM system prefill failed')
        start = time.perf_counter()
        if not self.lib.omni_c_stream_prefill(self.ctx, row['audio'].encode(), 1):
            raise RuntimeError('MiniCPM audio prefill failed')
        if not self.lib.omni_c_stream_decode(self.ctx, str(self.work).encode(), 0):
            raise RuntimeError('MiniCPM decode failed')
        buf, chunks = ctypes.create_string_buffer(65536), []
        while self.lib.omni_c_pop_text(self.ctx, buf, len(buf)):
            chunk = buf.value.decode(errors='replace')
            if chunk not in ('__END_OF_TURN__','__IS_LISTEN__'):
                chunks.append(chunk)
        text_ms = 1000*(time.perf_counter()-start)
        if self.tts:
            self.reset_safe = bool(self.lib.omni_c_drain_audio(self.ctx, 30000, 1000))
            if not self.reset_safe:
                raise RuntimeError('MiniCPM TTS drain timed out; context cannot safely reset')
        # Preserve token-fragment spacing exactly; do not invent separators.
        return ''.join(chunks).strip(), {'audio_to_text_ms': text_ms,
                                        'tts_drain_in_request': self.tts}

    def close(self):
        self.lib.omni_c_free(self.ctx)


class Jaeger:
    def __init__(self, work):
        from jaeger_agent_omni import converse
        self.ear = converse.load_trained_semantic('quasi')
        node = ROOT/'jaeger_agent_omni/nodes/jaegeromni-41m-medium'
        self.identity = {'node': node.name, 'adapter_sha256': sha(node/'weights/adapter.npz'),
                         'runtime': 'MLX production converse endpointed ear', 'tts': False}

    def infer(self, row, system, max_tokens):
        import soundfile as sf
        waveform, _ = sf.read(row['audio'], dtype='float32')
        if row['task'] == 'asr':
            self.ear.max_tokens = max_tokens
            return self.ear.transcribe(waveform)
        return self.ear.respond(waveform, system=system, history=[], max_tokens=max_tokens)

    def close(self):
        pass


def score(row, raw):
    from jaeger_agent_omni.converse import parse_heard_reply
    if row['task'] == 'answer':
        hypothesis, answer = parse_heard_reply(raw)
        valid = 'HEARD:' in raw and 'REPLY:' in raw
        correct = valid and graded(answer, row['answers'])
    else:
        hypothesis, answer, valid, correct = raw, None, True, None
    reference = normalize(row['reference'])
    edits = round(wer(reference, normalize(hypothesis))*len(reference))
    return {'hypothesis': hypothesis, 'answer': answer, 'format_valid': valid,
            'correct': correct, 'edits': edits, 'reference_words': len(reference)}


def summarize(rows):
    out = {}
    for corpus in sorted({r['corpus'] for r in rows}):
        group = [r for r in rows if r['corpus'] == corpus]
        words = sum(r['reference_words'] for r in group)
        out[corpus] = {'n': len(group), 'wer': sum(r['edits'] for r in group)/max(1,words),
                       'reference_words': words, 'edits': sum(r['edits'] for r in group),
                       'errors': sum(bool(r['error']) for r in group),
                       'request_ms_median': statistics.median(r['request_ms'] for r in group),
                       'request_ms_p95': float(np.percentile([r['request_ms'] for r in group],95))}
        if corpus == 'cases12':
            out[corpus]['correct'] = sum(bool(r['correct']) for r in group)
    return out


def run(args):
    from training.context_distill import refuse_gpu_contention
    from jaeger_agent_omni.converse import NN_SYSTEM
    refuse_gpu_contention()
    manifest = json.loads(args.manifest.read_text())
    cases = [r for r in manifest['cases'] if args.corpus == 'all' or r['corpus'] == args.corpus]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError('empty selection')
    for row in cases:
        if sha(row['audio']) != row['audio_sha256']:
            raise ValueError('fixture changed: '+row['id'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work = ROOT/'artifacts/paired-omni'/args.output.stem
    model = MiniCPM(work, args.minicpm_template) if args.model == 'minicpm' else Jaeger(work)
    rows = []
    try:
        # Warm one clip of every selected task; all scored turns use fresh state.
        for task in sorted({r['task'] for r in cases}):
            row = next(r for r in cases if r['task'] == task)
            model.infer(row, ASR_SYSTEM if task == 'asr' else NN_SYSTEM, 128 if task == 'asr' else 140)
        progress = args.output.with_suffix('.rows.jsonl')
        progress.write_text('')
        for index, row in enumerate(cases):
            start, error, raw, metrics = time.perf_counter(), None, '', {}
            try:
                raw, metrics = model.infer(row, ASR_SYSTEM if row['task'] == 'asr' else NN_SYSTEM,
                                           128 if row['task'] == 'asr' else 140)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            result = {**row, 'raw': raw, 'error': error, 'metrics': metrics,
                      'request_ms': 1000*(time.perf_counter()-start), **score(row,raw)}
            rows.append(result)
            with progress.open('a') as f:
                f.write(json.dumps(result)+'\n')
            print(f"[{index+1}/{len(cases)}] {row['id']} edits={result['edits']} correct={result['correct']} {raw[:100]!r}",flush=True)
    finally:
        model.close()
    report = {'schema': 1, 'model': args.model, 'identity': model.identity,
              'manifest_sha256': sha(args.manifest), 'code_sha256': sha(__file__),
              'scorer_sha256': {n: sha(ROOT/n) for n in ('training/evaluate.py','research/grading.py')},
              'prompts': {'asr': ASR_SYSTEM,'answer':NN_SYSTEM}, 'max_tokens': {'asr':128,'answer':140},
              'command': [sys.executable,*sys.argv], 'host': platform.platform(),
              'rss_peak_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'memory_scope': 'process peak RSS, not total unified/device memory; no memory superiority claim',
              'latency_scope': ('fresh request setup and system prefill through complete text; '
                                'includes TTS drain only for user-tts; '
                                'excludes model load, warmup and physical endpointing'),
              'population': manifest['population'], 'summary': summarize(rows), 'rows':rows,
              'full_system_comparison_complete': False}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary']),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--model',choices=['jaeger','minicpm'],default='jaeger')
    p.add_argument('--minicpm-template',choices=['system','user','user-tts'],default='system',
                   help='explicit prompt validation probe; user-tts uses native speech generation')
    p.add_argument('--cache',type=Path,default=CACHE)
    p.add_argument('--manifest',type=Path,default=ROOT/'research/candidate_evals/paired-omni-fixtures-20260907.json')
    p.add_argument('--corpus',choices=['all','devclean','ami','cases12'],default='all')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.action=='run' and args.output is None: p.error('run requires --output')
    (prepare if args.action=='prepare' else run)(args)


if __name__=='__main__': main()
