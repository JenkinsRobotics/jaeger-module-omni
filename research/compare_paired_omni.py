"""Reject incompatible benchmarks before comparing measured quality/latency."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def compare(jaeger, minicpm):
    for key in ('manifest_sha256','scorer_sha256','prompts','max_tokens',
                'host_name','cpu','host_ram_bytes'):
        if not jaeger.get(key) or jaeger.get(key) != minicpm.get(key):
            raise ValueError('benchmark mismatch or missing evidence: '+key)
    if jaeger.get('model') != 'jaeger' or minicpm.get('model') != 'minicpm':
        raise ValueError('expected Jaeger and MiniCPM reports in that order')
    if (jaeger['identity'].get('tts') is not False or
            minicpm['identity'].get('tts') is not False or
            minicpm['identity'].get('audio_generation_template') is not True):
        raise ValueError('requires qualified audio template and text-only output on both engines')
    if len(jaeger['rows']) != len(minicpm['rows']):
        raise ValueError('different row counts')
    for a,b in zip(jaeger['rows'],minicpm['rows']):
        for key in ('id','audio_sha256','reference','task'):
            if a[key] != b[key]: raise ValueError('fixture mismatch: '+key)
    results={}
    for corpus,a in jaeger['summary'].items():
        b=minicpm['summary'][corpus]
        if (a['n'],a['reference_words']) != (b['n'],b['reference_words']):
            raise ValueError('different scoring populations')
        results[corpus]={'n':a['n'],'reference_words':a['reference_words'],
                         'jaeger_wer':a['wer'],'minicpm_wer':b['wer'],
                         'jaeger_edits':a['edits'],'minicpm_edits':b['edits'],
                         'jaeger_request_median_ms':a['request_ms_median'],
                         'minicpm_request_median_ms':b['request_ms_median'],
                         'jaeger_request_p95_ms':a['request_ms_p95'],
                         'minicpm_request_p95_ms':b['request_ms_p95'],
                         'errors':{'jaeger':a['errors'],'minicpm':b['errors']}}
        if 'correct' in a:
            results[corpus].update(jaeger_correct=a['correct'],minicpm_correct=b['correct'])
    return {'matched_inputs_and_scorers':True,'results':results,
            'hardware':{k:jaeger[k] for k in ('host_name','cpu','host_ram_bytes')},
            'population':jaeger['population'],
            'latency_scope':'fresh request through complete text; model load/warmup, speech synthesis and physical endpointing excluded',
            'prompt_scope':'same task instruction, each engine uses its native audio template',
            'full_system_superiority_established':False,
            'missing_axes':['visual QA','action outcomes against MiniCPM','physical interruptions','spoken response quality','complete device memory']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jaeger',type=Path,required=True)
    p.add_argument('--minicpm',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=compare(json.loads(args.jaeger.read_text()),json.loads(args.minicpm.read_text()))
    result.update(command=[sys.executable,*sys.argv],
                  inputs={str(v):hashlib.sha256(v.read_bytes()).hexdigest() for v in (args.jaeger,args.minicpm)},
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['results'],indent=2))


if __name__=='__main__':main()
