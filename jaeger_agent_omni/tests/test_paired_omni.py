import unittest
from research.paired_omni_benchmark import score, summarize


class PairedOmniTests(unittest.TestCase):
    def test_shared_wer_counts_insertions_and_deletions(self):
        case={'task':'asr','reference':'one two three'}
        r=score(case,'one four three five')
        self.assertEqual(r['edits'],2)
        self.assertEqual(r['reference_words'],3)
        self.assertEqual(score(case,'')['edits'],3)

    def test_answer_requires_real_reply_field(self):
        case={'task':'answer','reference':'What is two plus two?', 'answers':['four']}
        self.assertFalse(score(case,'HEARD: four')['correct'])
        self.assertTrue(score(case,'HEARD: What is two plus two?\nREPLY: Four.')['correct'])

    def test_summary_pools_reference_words(self):
        rows=[dict(corpus='devclean',edits=1,reference_words=1,error=None,request_ms=10),
              dict(corpus='devclean',edits=0,reference_words=9,error=None,request_ms=20)]
        self.assertEqual(summarize(rows)['devclean']['wer'],.1)
