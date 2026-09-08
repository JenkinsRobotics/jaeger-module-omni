import threading
import unittest
from jaeger_agent_omni.core.runtime.speech_delivery import SpeechDelivery


class DeliveryTest(unittest.TestCase):
    def test_inflight_synthesis_cannot_reenter_after_barge(self):
        d = SpeechDelivery()
        key = d.begin()
        self.assertTrue(d.enqueue(key))
        synthesis_started, resume = threading.Event(), threading.Event()
        played = []
        def worker():
            synthesis_started.set()
            resume.wait(1)
            with d.lock:
                if d.valid(key): played.append(key)
        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(synthesis_started.wait(1))
        d.cancel_pending()
        resume.set(); thread.join(1)
        self.assertEqual(played, [])
        self.assertFalse(d.enqueue(key))
        d.close(key)
        self.assertEqual(d.summary()['completed_uninterrupted_replies'], 0)

    def test_completion_requires_all_playback_and_closed_generation(self):
        d = SpeechDelivery(); key = d.begin()
        d.enqueue(key); d.enqueue(key); d.played(key)
        d.close(key)
        self.assertEqual(d.summary()['completed_uninterrupted_replies'], 0)
        d.played(key)
        self.assertEqual(d.summary()['completed_uninterrupted_replies'], 1)
        d.cancel_pending()
        self.assertEqual(d.summary()['completed_uninterrupted_replies'], 1)

    def test_stopped_clause_is_not_completed_and_next_reply_is_valid(self):
        d = SpeechDelivery(); old = d.begin(); d.enqueue(old)
        d.cancel_pending(); d.played(old); d.close(old)
        new = d.begin(); self.assertTrue(d.enqueue(new))
        d.played(new); d.close(new)
        summary = d.summary()
        self.assertEqual(summary['played_clauses'], 1)
        self.assertEqual(summary['interrupted_replies'], 1)
        self.assertEqual(summary['completed_uninterrupted_replies'], 1)

    def test_failure_and_shutdown_never_count_as_finished(self):
        d = SpeechDelivery(); failed = d.begin(); d.enqueue(failed)
        d.fail(failed, 'output disconnected'); d.close(failed)
        stopped = d.begin(); d.enqueue(stopped); d.cancel_pending('shutdown')
        d.played(stopped); d.close(stopped)
        summary = d.summary()
        self.assertEqual(summary['failed_replies'], 1)
        self.assertEqual(summary['completed_uninterrupted_replies'], 0)
