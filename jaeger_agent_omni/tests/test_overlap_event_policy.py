import unittest
import numpy as np
from research.overlap_event_policy import alarm_onsets, event_score


class AlarmPolicyTest(unittest.TestCase):
    def test_early_alarm_cannot_receive_later_overlap_credit(self):
        prob = np.ones(12)
        available = np.repeat([.16, .32, .48], 4)
        labels = np.r_[np.zeros(4), np.ones(8)]
        # Earliest onset .16 is valid here; move the event to .20 to expose
        # the optimistic legacy rule that credits any interval intersection.
        labels[4] = 0
        score = event_score([(prob, labels, available)], .5, 1, 1)
        self.assertEqual(score['detected_events'], 0)
        self.assertEqual(score['false_events'], 1)

    def test_dwell_is_charged_at_actual_publication_time(self):
        available = np.repeat([.16, .32, .48], 4)
        prob = np.ones(12)
        self.assertEqual(alarm_onsets(prob, available, .5, 2, 1), [.32])
        score = event_score([(prob, np.ones(12), available)], .5, 2, 1)
        self.assertEqual(score['detected_events'], 1)
        self.assertAlmostEqual(score['delay_p95_ms'], 320)
