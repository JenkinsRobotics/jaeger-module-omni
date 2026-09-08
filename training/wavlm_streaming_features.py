"""Past-only WavLM windows: 1.28 s context, decisions every 160 ms.

Each group of four 40 ms features becomes available at the same window end.
Whisper, its adapter and Gemma are never loaded. CPU probe measures cost before
spending a GPU window on a complete cache.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.pretrained_overlap import MODEL, BASE_CACHE, digest

RATE, WINDOW, HOP = 16000, 20480, 2560


def windows(audio, frames):
    """Zero left padding is past silence; never read past a window deadline."""
    for first in range(0, frames, 4):
        count = min(4,frames-first)
        end = min(len(audio), (first+count)*640)
        chunk = np.zeros(WINDOW, np.float32)
        available = audio[max(0,end-WINDOW):end]
        chunk[-len(available):] = available
        yield first, count, end/RATE, chunk


def target_centers(first, available):
    # Partial final blocks retain their true centers instead of shifting their
    # features backwards by the missing 40 ms bins. Unused slots are discarded.
    return WINDOW/RATE - available + (first+np.arange(4)+.5)*.04


@torch.no_grad()
def encode(model, batch, device, targets=None):
    x = torch.from_numpy(np.stack(batch)).to(device)
    x = (x-x.mean(-1,keepdim=True))/torch.sqrt(x.var(-1,unbiased=False,keepdim=True)+1e-7)
    out = model(x,output_hidden_states=True)
    features = torch.stack([out.hidden_states[i] for i in (4,8,12)]).mean(0).cpu().numpy()
    source = (np.arange(features.shape[1])*320+200)/RATE
    if targets is None:
        targets = [WINDOW/RATE - HOP/RATE + (np.arange(4)+.5)*.04]*len(batch)
    return np.stack([np.stack([np.interp(target,source,f[:,i]) for i in range(768)],axis=1)
                     for f,target in zip(features,targets)]).astype(np.float16)


def probe(args):
    import soundfile as sf
    from transformers import WavLMModel
    torch.set_num_threads(2)
    model = WavLMModel.from_pretrained(MODEL,local_files_only=True).eval().to('cpu')
    audio,rate = sf.read(args.audio,dtype='float32')
    if rate != RATE or audio.ndim != 1: raise ValueError('requires mono 16 kHz')
    chunks = list(windows(audio, min(int(len(audio)/640),args.frames)))
    encode(model,[chunks[0][3]],'cpu')
    times = []
    for first,count,available,chunk in chunks:
        start = time.perf_counter(); result = encode(model,[chunk],'cpu',[target_centers(first,available)])
        times.append({'first_frame':first,'available_at_s':available,
                      'compute_ms':1000*(time.perf_counter()-start)})
        if result.shape != (1,4,768): raise ValueError(result.shape)
    report = {'command':[sys.executable,*sys.argv],'device':'cpu','torch_threads':2,
              'model':json.loads((MODEL/'SOURCE.json').read_text()),
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'audio_sha256':hashlib.sha256(args.audio.read_bytes()).hexdigest(),
              'context_ms':1280,'hop_ms':160,'max_buffering_ms':160,
              'compute_p95_ms':float(np.percentile([r['compute_ms'] for r in times],95)),
              'timings':times,'scope':'compute probe, no detector accuracy or physical delay claim'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='timings'}),flush=True)


def cache(args):
    import os
    import soundfile as sf
    from transformers import WavLMModel
    from training.context_distill import refuse_gpu_contention
    if args.device == 'mps': refuse_gpu_contention()
    torch.set_num_threads(2)
    model = WavLMModel.from_pretrained(MODEL,local_files_only=True).eval().to(args.device)
    rows = [json.loads(s) for s in (BASE_CACHE/'manifest.jsonl').read_text().splitlines()]
    if args.limit: rows = rows[:args.limit]
    args.cache.mkdir(parents=True,exist_ok=True)
    provenance = {'feature_mode':'bounded_past_1280ms_hop160ms', 'context_ms':1280,
                  'hop_ms':160,'max_buffering_ms':160,'layers':[4,8,12],
                  'pretrained':json.loads((MODEL/'SOURCE.json').read_text()),
                  'weights_sha256':digest(MODEL/'pytorch_model.bin'),
                  'source_manifest_sha256':digest(BASE_CACHE/'manifest.jsonl'),
                  'feature_code_sha256':digest(__file__)}
    marker=args.cache/'provenance.json'
    if marker.exists() and json.loads(marker.read_text()) != provenance:
        raise ValueError('bounded cache identity changed; choose a new cache directory')
    marker.write_text(json.dumps(provenance,indent=2)+'\n')
    output_rows=[];timings=[]
    for index,row in enumerate(rows):
        target=args.cache/Path(row['archive']).name
        if not target.exists():
            with np.load(row['archive']) as z: labels=z['labels'].copy()
            audio,rate=sf.read(row['audio'],dtype='float32')
            if rate!=RATE or audio.ndim!=1:raise ValueError('expected mono 16 kHz')
            chunks=list(windows(audio,len(labels)))
            features=np.empty((len(labels),768),np.float16)
            available_at=np.empty(len(labels),np.float32)
            started=time.perf_counter()
            for offset in range(0,len(chunks),args.batch):
                batch=chunks[offset:offset+args.batch]
                values=encode(model,[c[3] for c in batch],args.device,
                              [target_centers(c[0],c[2]) for c in batch])
                for (first,count,available,_),value in zip(batch,values):
                    features[first:first+count]=value[:count]
                    available_at[first:first+count]=available
            timings.append({'id':row['id'],'windows':len(chunks),
                            'seconds':time.perf_counter()-started,'audio_seconds':len(audio)/RATE})
            tmp=target.with_suffix('.partial.npz')
            np.savez(tmp,features=features,labels=labels,available_at=available_at)
            os.replace(tmp,target)
        output_rows.append({**row,'archive':str(target)})
        if args.limit or (index+1)%100==0:
            print(json.dumps({'cached':index+1,'total':len(rows),
                              'last_seconds':timings[-1]['seconds'] if timings else None}),flush=True)
    manifest=args.cache/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in output_rows))
    report={**provenance,'command':[sys.executable,*sys.argv],
            'output_manifest_sha256':digest(manifest),'rows':len(rows),'batch':args.batch,
            'device':args.device,'new_cache_timing':timings,
            'timing_scope':'batched cache throughput, not live per-decision cost',
            'runtime_policy_changed':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['probe','cache'])
    p.add_argument('--audio',type=Path,default=ROOT/'research/cases12/q01.wav')
    p.add_argument('--frames',type=int,default=80)
    p.add_argument('--cache',type=Path,default=Path.home()/'models/jaeger-training-data/overlap-wavlm-bounded-v1')
    p.add_argument('--device',choices=['cpu','mps'],default='mps')
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.batch < 1 or args.frames < 1: p.error('batch and frames must be positive')
    (probe if args.action=='probe' else cache)(args)


if __name__=='__main__': main()
