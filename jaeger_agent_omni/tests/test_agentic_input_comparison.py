import copy
import unittest

from research.compare_agentic_inputs import compare


def pair():
    a = {'input_mode': 'text', 'adapter_sha256': 'model', 'case_sources_sha256': 'cases',
         'initial_tools': ['write_file'], 'system_sha256': 'system', 'case_timeout_s': 20,
         'summary': {'verified_file_outcomes': {'contract_sha256': 'contract'}},
         'rows': [{'lane': 'a', 'id': 'file', 'audio_sha256': 'audio',
                   'prompt_fed_as_text': 'write note', 'case_pass': True,
                   'outcome': {'pass': False}}]}
    b = copy.deepcopy(a)
    b['input_mode'] = 'speech'
    b['rows'][0]['prompt_ground_truth_not_fed'] = b['rows'][0].pop('prompt_fed_as_text')
    return a, b


class InputComparisonTest(unittest.TestCase):
    def test_routing_success_does_not_become_outcome_success(self):
        result = compare(*pair())['summary']['speech']
        self.assertEqual(result['original_passed'], 1)
        self.assertEqual(result['verified_outcome_passed'], 0)

    def test_changed_request_cannot_enter_control(self):
        a, b = pair()
        b['rows'][0]['prompt_ground_truth_not_fed'] = 'read note'
        with self.assertRaisesRegex(ValueError, 'different request'):
            compare(a, b)

    def test_changed_outcome_contract_is_rejected(self):
        a, b = pair()
        b['summary']['verified_file_outcomes']['contract_sha256'] = 'weaker'
        with self.assertRaisesRegex(ValueError, 'outcome contracts'):
            compare(a, b)
