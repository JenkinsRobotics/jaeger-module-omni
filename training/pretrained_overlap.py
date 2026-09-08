"""Frozen WavLM + independent overlap head. Offline mode is an upper-bound
representation diagnostic, never a streaming promotion. No Gemma/Whisper use.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from training.parallel_overlap import CACHE as BASE_CACHE, digest, metrics, load_rows
from training.context_distill import refuse_gpu_contention

MODEL=Path.home()/'models/jaeger-training-data/pretrained/wavlm-base-plus'
CACHE=Path.home()/'models/jaeger-training-data/overlap-wavlm-offline-v1'


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(768)
        self.project=nn.Linear(768,128)
        self.gru=nn.GRU(128,128,batch_first=True)
        self.output=nn.Linear(128,1)

    def forward(self,x):
        h=F.gelu(self.project(self.norm(x)))
        h,_=self.gru(h)
        return self.output(h).squeeze(-1)


@torch.no_grad()
def extract(model,audio,frames,device):
    """Full-utterance normalization/attention, explicitly noncausal."""
    x=torch.as_tensor(audio,device=device).float()
    x=(x-x.mean())/torch.sqrt(x.var(unbiased=False)+1e-7)
    out=model(x[None],output_hidden_states=True)
    feature=torch.stack([out.hidden_states[i] for i in (4,8,12)]).mean(0)[0].cpu().numpy()
    # WavLM conv receptive field 400 samples, stride 320; label bins are 40 ms.
    source_t=(np.arange(len(feature))*320+200)/16000
    target_t=(np.arange(frames)+.5)*.04
    values=np.stack([np.interp(target_t,source_t,feature[:,i]) for i in range(768)],axis=1)
    return values.astype(np.float16)


def cache(args):
    import soundfile as sf
    from transformers import WavLMModel
    refuse_gpu_contention()
    torch.set_num_threads(2)
    model=WavLMModel.from_pretrained(args.model,local_files_only=True).eval().to(args.device)
    for p in model.parameters(): p.requires_grad_(False)
    original=[json.loads(s) for s in (BASE_CACHE/'manifest.jsonl').read_text().splitlines()]
    if args.limit: original=original[:args.limit]
    args.cache.mkdir(parents=True,exist_ok=True)
    rows=[];timings=[]
    weights=next(args.model.glob('*.safetensors'),None) or next(args.model.glob('pytorch_model.bin'))
    provenance={'source_manifest_sha256':digest(BASE_CACHE/'manifest.jsonl'),
                'pretrained':json.loads((args.model/'SOURCE.json').read_text()),
                'weights_sha256':digest(weights),'layers':[4,8,12],
                'feature_mode':'offline_full_utterance_bidirectional',
                'code_sha256':digest(__file__)}
    marker=args.cache/'provenance.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if any(old[k]!=provenance[k] for k in ('weights_sha256','source_manifest_sha256','layers','feature_mode','code_sha256')):
            raise ValueError('cache identity mismatch; use a new cache directory')
    marker.write_text(json.dumps(provenance,indent=2)+'\n')
    for i,row in enumerate(original):
        target=args.cache/Path(row['archive']).name
        with np.load(row['archive']) as z: labels=z['labels'].copy()
        if not target.exists():
            audio,rate=sf.read(row['audio'],dtype='float32')
            if rate!=16000 or audio.ndim!=1: raise ValueError('expected mono 16 kHz')
            started=time.perf_counter()
            features=extract(model,audio,len(labels),args.device)
            elapsed=time.perf_counter()-started
            timings.append({'id':row['id'],'seconds':elapsed,'audio_seconds':len(audio)/rate})
            np.savez(target,features=features,labels=labels)
        rows.append({**row,'archive':str(target)})
        if (i+1)%100==0 or args.limit:
            print(json.dumps({'cached':i+1,'total':len(original),'last_seconds':timings[-1]['seconds'] if timings else None}),flush=True)
    manifest=args.cache/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report={**provenance,'command':[sys.executable,*sys.argv],
            'output_manifest_sha256':digest(manifest),'rows':len(rows),
            'encoder_parameters':sum(p.numel() for p in model.parameters()),
            'counts':{s:sum(r['split']==s for r in rows) for s in ('train','validation','test')},
            'new_cache_timing':timings,'streaming':False,'runtime_policy_changed':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


@torch.no_grad()
def predict(model,rows,device):
    model.eval()
    return [(torch.sigmoid(model(torch.from_numpy(x[None]).to(device)))[0].cpu().numpy(),y)
            for _,x,y in rows]


def train(args):
    refuse_gpu_contention();torch.set_num_threads(2);torch.manual_seed(args.seed)
    data=load_rows(args.cache)
    for s in ('train','validation','test'):
        if not data.get(s):raise ValueError('missing split '+s)
    groups={s:{v for row,_,_ in rows for v in row['source_ids']} for s,rows in data.items()}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if groups[a]&groups[b]:raise ValueError('source leakage')
    model=Head().to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    rng=np.random.default_rng(args.seed)
    history=[];best=-1;started=time.perf_counter()
    args.checkpoint.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,args.epochs+1):
        model.train();losses=[];order=rng.permutation(len(data['train']))
        for offset in range(0,len(order),args.batch):
            rows=[data['train'][i] for i in order[offset:offset+args.batch]]
            length=max(len(y) for _,_,y in rows)
            x=np.zeros((len(rows),length,768),np.float32)
            y=np.zeros((len(rows),length),np.float32);mask=y.copy()
            for i,(_,features,labels) in enumerate(rows):
                x[i,:len(labels)]=features;y[i,:len(labels)]=labels;mask[i,:len(labels)]=1
            logits=model(torch.from_numpy(x).to(args.device))
            valid=torch.from_numpy(mask).to(args.device)
            loss=(F.binary_cross_entropy_with_logits(logits,torch.from_numpy(y).to(args.device),reduction='none')*valid).sum()/valid.sum()
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();losses.append(float(loss.item()))
        val=predict(model,data['validation'],args.device)
        scores=[]
        for threshold in np.linspace(.1,.9,33):
            by_corpus={c:metrics([p for p,r in zip(val,data['validation']) if r[0]['corpus']==c],float(threshold))
                       for c in ('bargein_train','ami_overlap')}
            scores.append({'threshold':float(threshold),'macro_f1':float(np.mean([m['f1'] for m in by_corpus.values()])),
                           'by_corpus':by_corpus})
        selected=max(scores,key=lambda x:x['macro_f1'])
        record={'epoch':epoch,'loss':float(np.mean(losses)),'validation':selected,'elapsed_seconds':time.perf_counter()-started}
        history.append(record);print(json.dumps(record),flush=True)
        if selected['macro_f1']>best:
            best=selected['macro_f1']
            torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'threshold':selected['threshold'],'epoch':epoch},args.checkpoint)
    saved=torch.load(args.checkpoint,map_location=args.device,weights_only=True)
    model.load_state_dict(saved['model'])
    provenance=json.loads((args.cache/'provenance.json').read_text())
    test={}; availability={}
    for c in ('bargein_train','ami_overlap'):
        selected_rows=[r for r in data['test'] if r[0]['corpus']==c]
        predictions=predict(model,selected_rows,args.device)
        test[c]=metrics(predictions,saved['threshold'])
        if provenance['feature_mode'].startswith('bounded_'):
            availability[c]=buffered_detection_delays(predictions,selected_rows,saved['threshold'])
    report={'command':[sys.executable,*sys.argv],'code_sha256':digest(__file__),
            'cache_provenance':provenance,
            'manifest_sha256':digest(args.cache/'manifest.jsonl'),
            'head_parameters':sum(p.numel() for p in model.parameters()),'history':history,
            'selected_epoch':saved['epoch'],'test':test,'checkpoint_sha256':digest(args.checkpoint),
            'buffered_detection':availability,
            'frame_gate_pass':all(s['f1']>=.81 for s in test.values()),
            'promotion_pass':False,
            'reason':('offline encoder uses future audio; causal bounded-context evaluation required'
                      if not availability else 'bounded encoder; actual compute and physical policy gates remain required'),
            'pretraining_overlap_audit':'supervised splits disjoint; pretraining identity overlap not audited',
            'runtime_policy_changed':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'test':test,'promotion_pass':False}),flush=True)


def buffered_detection_delays(predictions, rows, threshold):
    """Retain the original event matching, but charge actual feature availability.

    These are still optimistic about compute and early-onset false alarms;
    they cannot qualify live policy by themselves.
    """
    from training.parallel_overlap import intervals
    delays=[]
    for (prob,labels),(row,_,_) in zip(predictions,rows):
        with np.load(row['archive']) as z: available=z['available_at']
        truth=intervals(labels>=.5); used=set()
        for start,end in intervals(prob>=threshold):
            matches=[i for i,(a,b) in enumerate(truth)
                     if i not in used and max(a,start)<min(b,end)]
            if matches:
                i=matches[0];used.add(i)
                delays.append(max(0,float(available[start])-truth[i][0]*.04)*1000)
    return {'detected_events':len(delays),
            'p95_ms':float(np.percentile(delays,95)) if delays else None,
            'includes_buffering':True,'includes_compute':False,
            'event_matching':'unchanged legacy overlap intervals; early-onset alarms not separately penalized'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['cache','train'])
    p.add_argument('--model',type=Path,default=MODEL)
    p.add_argument('--cache',type=Path,default=CACHE)
    p.add_argument('--checkpoint',type=Path,default=REPO/'artifacts/overlap-wavlm-offline-v1.pt')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--device',default='mps',choices=['cpu','mps'])
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=7)
    args=p.parse_args();(cache if args.action=='cache' else train)(args)


if __name__=='__main__':main()
