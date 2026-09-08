"""State-based checks declared before the matched text/speech corpus runs."""
import re


def check_outcome(contract, before, after, answer, error=None):
    """Check actual sandbox state, never infer an outcome from requested tools."""
    paths = ['skills/' + contract['file'], 'workspace/' + contract['file']]
    prior = {p: before[p] for p in paths if p in before}
    current = {p: after[p] for p in paths if p in after}
    checks = {'no_error': error is None}
    operation = contract['operation']
    if operation == 'delete':
        checks.update(existed_before=bool(prior), absent_after=not current)
    else:
        checks['one_file_exists'] = len(current) == 1
        content = next(iter(current.values()), {}).get('content', '').strip()
        if contract.get('numeric'):
            content = re.sub(r'(?<=\d),(?=\d)', '', content)
        if 'content' in contract:
            checks['file_content'] = content == contract['content']
        if 'lines' in contract:
            checks['file_lines'] = [s.strip() for s in content.splitlines()] == contract['lines']
        if operation == 'read':
            checks['existed_unchanged'] = bool(prior) and prior == current
    for i, expected in enumerate(contract.get('answer_contains', [])):
        checks[f'answer_{i}'] = expected.casefold() in str(answer).casefold()
    return {'pass': all(checks.values()), 'checks': checks,
            'observed_before': prior, 'observed_after': current,
            'success_claim_review_required': not all(checks.values())}
