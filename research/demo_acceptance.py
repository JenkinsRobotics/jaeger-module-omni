"""Evaluate the frozen demo contract; missing evidence never passes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text()) if path else {}


def physical_checks(report):
    m = report.get('metrics') or {}
    speech = m.get('speech_delivery') or {}
    return {'physical_capture': report.get('phase') == 'physical-room',
            'thirty_minutes': float(m.get('fed_seconds', 0)) >= 1800,
            'zero_loop_drops': m.get('dropped_seconds') == 0,
            'zero_capture_overflows': m.get('mic_queue_overflow_blocks') == 0,
            'no_runtime_errors': report.get('errors') == [],
            'uninterrupted_speech': speech.get('completed_uninterrupted_replies', 0) >= 1,
            'no_speech_failures': speech.get('failed_replies') == 0,
            'tools_exercised': (m.get('agentic') or {}).get('tool_calls', 0) >= 3}


def assess(suite, answers, actions, vision, campaigns, physical, review=None,
           action_sha=None, physical_shas=None):
    answer_rows = {r['id']:r for r in answers.get('rows', [])}
    action_rows = {r['lane']+':'+r['id']:r for r in actions.get('rows', [])}
    visual_rows = {r['name']:r for r in vision.get('checks', [])}
    fixture_sha = suite['input_sha256'].get('research/candidate_evals/paired-omni-fixtures-20260907.json')
    answers_matched = answers.get('model') == 'jaeger' and answers.get('manifest_sha256') == fixture_sha
    rows = []
    for case in suite['cases']:
        key = case['key']; passed = False
        if case['kind'] == 'answer':
            row = answer_rows.get(key, {})
            passed = answers_matched and row.get('correct') is True and not row.get('error')
        elif case['kind'] == 'action':
            row = action_rows.get(key, {})
            passed = ((row.get('outcome') or {}).get('pass') is True and row.get('input_mode') == 'speech'
                      and row.get('audio_sha256') == case.get('audio_sha256'))
        elif case['kind'] == 'vision':
            passed = vision.get('gate_valid') is True and visual_rows.get(key, {}).get('passed') is True
        else:
            campaign = campaigns.get(case['session'], {})
            check = next((r for r in campaign.get('checks', []) if r['name'] == key), {})
            passed = campaign.get('gate_valid') is True and check.get('passed') is True
        rows.append({**case, 'passed': bool(passed)})
    review = review or {}
    reviewed = review.get('cases', {})
    action_keys = [r['key'] for r in suite['cases'] if r['kind'] == 'action']
    no_fabrication = (action_sha is not None and review.get('action_evidence_sha256') == action_sha
                      and all(reviewed.get(k, {}).get('fabricated_success') is False
                              and reviewed[k].get('reason') for k in action_keys))
    # A tool-call counter proves dispatch, not a completed physical action.
    # Require the same evidence-bound outcome/claim review for the room runs.
    room_reviews = review.get('physical_runs', {})
    physical_shas = physical_shas or []
    room_outcomes = (len(physical_shas) == len(physical) == 3 and
        all(room_reviews.get(sha, {}).get('fabricated_success') is False
            and room_reviews[sha].get('verified_action_outcomes', 0) >= 3
            and room_reviews[sha].get('reason') for sha in physical_shas))
    physical_rows = [physical_checks(p) for p in physical]
    times = [(p.get('started_at', ''), p.get('finished_at', '')) for p in physical]
    chronological = bool(times) and all(a and b and a < b for a,b in times)
    chronological = chronological and all(times[i][1] <= times[i+1][0] for i in range(len(times)-1))
    identities = [json.dumps({'model':p.get('model'), 'runtime':p.get('runtime_sha256'),
                              'configuration':p.get('configuration')}, sort_keys=True) for p in physical]
    same_setup = bool(identities) and len(set(identities)) == 1 and all(p.get('runtime_sha256') for p in physical)
    rate = sum(r['passed'] for r in rows) / max(1, len(rows))
    gates = {'completion_at_least_95_percent': rate >= suite['completion_min'],
             'no_fabricated_action_success': bool(no_fabrication and room_outcomes),
             'three_consecutive_physical_runs': len(physical_rows) == 3 and chronological and bool(same_setup)
                and bool(room_outcomes) and all(all(p.values()) for p in physical_rows)}
    return {'completion_rate': rate, 'passed_cases': sum(r['passed'] for r in rows),
            'total_cases': len(rows), 'cases':rows, 'physical_checks':physical_rows,
            'gates':gates, 'physical_action_outcomes_reviewed': bool(room_outcomes),
            'demo_acceptance_pass':all(gates.values()),
            'standing_release_gates_waived':False, 'minicpm_superiority_claim':False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', type=Path, default=ROOT/'research/demo_suite_v1.json')
    for name in ('answers','actions','vision','review'): p.add_argument('--'+name, type=Path)
    p.add_argument('--campaign', type=Path, action='append', default=[])
    p.add_argument('--physical', type=Path, action='append', default=[], help='native .live.json reports in run order')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    suite = load(args.suite)
    for path, expected in suite['input_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != expected:
            raise ValueError('frozen demo input changed: '+path)
    campaigns = [load(v) for v in args.campaign]
    report = assess(suite, load(args.answers), load(args.actions), load(args.vision),
                    {d.get('session'): d for d in campaigns}, [load(v) for v in args.physical],
                    load(args.review), hashlib.sha256(args.actions.read_bytes()).hexdigest() if args.actions else None,
                    [hashlib.sha256(v.read_bytes()).hexdigest() for v in args.physical])
    paths = [v for v in (args.suite,args.answers,args.actions,args.vision,args.review) if v] + args.campaign + args.physical
    report.update(command=[sys.executable,*sys.argv],
                  evidence_sha256={str(v):hashlib.sha256(v.read_bytes()).hexdigest() for v in paths})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['gates']))
    return 0 if report['demo_acceptance_pass'] else 1


if __name__ == '__main__': raise SystemExit(main())
