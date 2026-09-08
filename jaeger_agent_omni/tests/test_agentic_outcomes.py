import unittest
from research.agentic_outcomes import check_outcome


class OutcomeTest(unittest.TestCase):
    def test_claim_without_artifact_fails(self):
        result = check_outcome({'file':'bench.txt', 'content':'hello', 'operation':'write'},
                               {}, {}, 'Done, I saved the file.')
        self.assertFalse(result['pass'])
        self.assertTrue(result['success_claim_review_required'])

    def test_wrong_content_fails_despite_file(self):
        result = check_outcome({'file':'math.txt', 'content':'1093', 'operation':'write'},
                               {}, {'skills/math.txt':{'content':'1092'}}, 'Saved.')
        self.assertFalse(result['pass'])

    def test_delete_requires_actual_prior_file(self):
        contract = {'file':'note.txt', 'operation':'delete'}
        self.assertFalse(check_outcome(contract, {}, {}, 'Deleted.')['pass'])
        self.assertTrue(check_outcome(contract, {'skills/note.txt':{'content':'x'}}, {}, 'Deleted.')['pass'])

    def test_read_checks_state_and_answer(self):
        contract = {'file':'note.txt', 'content':'penguins are birds',
                    'operation':'read', 'answer_contains':['penguins are birds']}
        files = {'skills/note.txt':{'content':'penguins are birds'}}
        self.assertFalse(check_outcome(contract, files, files, 'I read it.')['pass'])
        self.assertTrue(check_outcome(contract, files, files, 'It says penguins are birds.')['pass'])
