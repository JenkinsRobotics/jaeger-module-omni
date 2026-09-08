"""Validation-selected, availability-aware alarm policy for the bounded NN.

CPU only. The existing frame scorer is preserved. An alarm onset must actually
occur inside a truth event; an early alarm cannot receive later overlap credit.
This diagnostic excludes encoder compute and does not promote a runtime policy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def alarm_onsets(prob, available, threshold, dwell, release):
    """One decision per published block, using its newest frame's probability."""
    if len(prob) != len(available) or np.any(np.diff(available) < 0):
        raise ValueError('invalid feature availability')
    indices = np.flatnonzero(np.r_[np.diff(available) != 0, True]) if len(prob) else []
    active, high, low, alarms = False, 0, 0, []
    for i in indices:
        if prob[i] >= threshold:
            high += 1
            low = 0
            if not active and high >= dwell:
                active = True
                alarms.append(float(available[i]))
        else:
            high = 0
            low += 1
            if low >= release:
                active = False
    return alarms


def event_score(records, threshold, dwell, release):
    from training.parallel_overlap import intervals, metrics
    matched, false, total, seconds, delays = 0, 0, 0, 0.0, []
    for prob, labels, available in records:
        truth = [(a*.04, b*.04) for a, b in intervals(labels >= .5)]
        used = set()
        total += len(truth)
        seconds += len(labels)*.04
        for onset in alarm_onsets(prob, available, threshold, dwell, release):
            hits = [i for i, (a, b) in enumerate(truth) if i not in used and a <= onset < b]
            if hits:
                i = hits[0]
                used.add(i)
                matched += 1
                delays.append(1000*(onset-truth[i][0]))
            else:
                false += 1
    frame = metrics([(p, y) for p, y, _ in records], threshold)
    return {'frame_f1': frame['f1'], 'events': total, 'detected_events': matched,
            'event_recall': matched/max(1, total), 'false_events': false,
            'false_events_per_minute': false/max(1e-9, seconds/60),
            'delay_p95_ms': float(np.percentile(delays, 95)) if delays else None,
            'seconds': seconds}


def passes(score):
    return (score['frame_f1'] >= .81 and score['event_recall'] >= .90
            and score['false_events_per_minute'] <= 1
            and score['delay_p95_ms'] is not None and score['delay_p95_ms'] <= 300)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--training-report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    from training.pretrained_overlap import Head, load_rows, predict
    torch.set_num_threads(2)
    report = json.loads(args.training_report.read_text())
    if sha(args.checkpoint) != report['checkpoint_sha256']:
        raise ValueError('checkpoint does not match completed training report')
    if sha(args.cache/'manifest.jsonl') != report['manifest_sha256']:
        raise ValueError('feature manifest changed')
    if not report['cache_provenance']['feature_mode'].startswith('bounded_'):
        raise ValueError('requires bounded features with availability timestamps')
    data = load_rows(args.cache)
    model = Head()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True)['model'])
    corpora = ('bargein_train', 'ami_overlap')

    def records(split):
        values = {c: [] for c in corpora}
        predictions = predict(model, data[split], 'cpu')
        for (row, _, _), (prob, labels) in zip(data[split], predictions):
            with np.load(row['archive']) as z:
                available = z['available_at'].copy()
            values[row['corpus']].append((prob, labels, available))
        return values

    validation = records('validation')
    grid = []
    # Declared before test inference: 17 thresholds × 2 dwell × 2 release.
    for threshold in np.linspace(.1, .9, 17):
        for dwell in (1, 2):
            for release in (1, 2):
                scores = {c: event_score(validation[c], float(threshold), dwell, release) for c in corpora}
                grid.append({'threshold': float(threshold), 'dwell_blocks': dwell,
                             'release_blocks': release, 'scores': scores})

    def objective(row):
        scores = list(row['scores'].values())
        # Feasibility first. If infeasible, retain frame quality before event cost;
        # do not chase test recall by selecting an always-on classifier.
        return (all(passes(s) for s in scores),
                min(s['frame_f1'] for s in scores),
                min(s['event_recall'] for s in scores),
                -max(s['false_events_per_minute'] for s in scores))

    selected = max(grid, key=objective)
    print(json.dumps({'selected_validation': selected}), flush=True)
    test = records('test')
    scores = {c: event_score(test[c], selected['threshold'], selected['dwell_blocks'],
                             selected['release_blocks']) for c in corpora}
    result = {'command': [sys.executable, *sys.argv], 'code_sha256': sha(__file__),
              'training_report_sha256': sha(args.training_report),
              'checkpoint_sha256': sha(args.checkpoint), 'manifest_sha256': report['manifest_sha256'],
              'validation_grid': grid, 'selected': selected, 'test': scores,
              'offline_policy_gate_pass': all(passes(s) for s in scores.values()),
              'promotion_pass': False, 'production_changed': False,
              'scope': 'CPU inference; onset must fall inside truth interval; buffering included, encoder compute excluded',
              'limits': ['synthetic barge and AMI development data, not physical playback',
                         'frame F1 is unchanged per-frame threshold score, separate from alarm policy',
                         'validation selection reuses checkpoint-selection validation population']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'test': scores, 'promotion_pass': False}), flush=True)


if __name__ == '__main__':
    main()
