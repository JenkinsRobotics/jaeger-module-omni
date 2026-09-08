import unittest
from research.demo_acceptance import assess, physical_checks
from research.physical_omni_trial import actor_schedule


class DemoAcceptanceTest(unittest.TestCase):
    def test_missing_evidence_cannot_pass(self):
        suite = {'cases':[{'id':'a','kind':'answer','key':'a'}],
                 'input_sha256':{}, 'completion_min':.95}
        result = assess(suite, {}, {}, {}, {}, [])
        self.assertFalse(result['demo_acceptance_pass'])
        self.assertEqual(result['passed_cases'], 0)

    def test_generated_reply_is_not_proof_of_uninterrupted_speech(self):
        report = {'phase':'physical-room','errors':[], 'metrics':{
            'fed_seconds':1800,'dropped_seconds':0,'mic_queue_overflow_blocks':0,
            'completed_replies':100,'agentic':{'tool_calls':3}}}
        checks = physical_checks(report)
        self.assertFalse(checks['uninterrupted_speech'])
        self.assertFalse(checks['no_speech_failures'])

    def test_text_control_never_counts_as_speech_demo_success(self):
        suite = {'cases':[{'id':'action:a:x','kind':'action','key':'a:x','audio_sha256':'same'}],
                 'input_sha256':{}, 'completion_min':.95}
        actions = {'rows':[{'lane':'a','id':'x','input_mode':'text','audio_sha256':'same','outcome':{'pass':True}}]}
        self.assertEqual(assess(suite, {}, actions, {}, {}, [])['passed_cases'], 0)

    def test_long_physical_schedule_exercises_every_block(self):
        schedule = list(actor_schedule([(6,'hello'),(279,'bye')],1800,True))
        self.assertEqual(len(schedule),12)
        self.assertEqual(schedule[-1],(1779,'bye'))
        self.assertEqual(len(list(actor_schedule([(6,'hello')],1800,False))),1)

    def test_dispatch_counts_cannot_certify_physical_actions(self):
        suite = {'cases': [], 'input_sha256': {}, 'completion_min': .95}
        reports = [{'phase': 'physical-room', 'errors': [], 'metrics': {
            'fed_seconds': 1800, 'dropped_seconds': 0, 'mic_queue_overflow_blocks': 0,
            'speech_delivery': {'completed_uninterrupted_replies': 10, 'failed_replies': 0},
            'agentic': {'tool_calls': 100}}, 'started_at': str(i*2+1),
            'finished_at': str(i*2+2), 'runtime_sha256': {'live': 'same'}} for i in range(3)]
        self.assertTrue(all(all(physical_checks(p).values()) for p in reports))
        result = assess(suite, {}, {}, {}, {}, reports, physical_shas=['a', 'b', 'c'])
        self.assertFalse(result['gates']['three_consecutive_physical_runs'])
        self.assertFalse(result['physical_action_outcomes_reviewed'])
