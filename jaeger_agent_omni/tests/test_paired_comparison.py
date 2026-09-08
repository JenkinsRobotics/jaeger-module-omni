import copy
import unittest
from research.compare_paired_omni import compare


def pair():
    a={'model':'jaeger','manifest_sha256':'manifest','scorer_sha256':{'wer':'scorer'},
       'prompts':{'asr':'same'},'max_tokens':{'asr':128},'host_name':'same',
       'cpu':'same','host_ram_bytes':32,'identity':{'tts':False},'population':'development',
       'rows':[{'id':'x','audio_sha256':'audio','reference':'hello','task':'asr'}],
       'summary':{'clean':{'n':1,'reference_words':1,'wer':0,'edits':0,
                           'request_ms_median':1,'request_ms_p95':1,'errors':0}}}
    b=copy.deepcopy(a);b['model']='minicpm';b['identity']['audio_generation_template']=True
    return a,b


class ComparisonTest(unittest.TestCase):
    def test_identical_transcripts_do_not_excuse_different_audio(self):
        a,b=pair();b['rows'][0]['audio_sha256']='different'
        with self.assertRaisesRegex(ValueError,'fixture mismatch'):compare(a,b)

    def test_wrong_reference_template_is_rejected(self):
        a,b=pair();b['identity']['audio_generation_template']=False
        with self.assertRaisesRegex(ValueError,'qualified audio template'):compare(a,b)

    def test_different_hardware_cannot_enter_matched_latency_claim(self):
        a,b=pair();b['cpu']='different'
        with self.assertRaisesRegex(ValueError,'benchmark mismatch'):compare(a,b)

    def test_valid_pair_does_not_claim_full_system_superiority(self):
        result=compare(*pair())
        self.assertTrue(result['matched_inputs_and_scorers'])
        self.assertFalse(result['full_system_superiority_established'])
