import threading
import unittest
from unittest.mock import Mock

from jaeger_agent_omni.core.runtime.device_lifecycle import close_audio_stream


class DeviceLifecycleTest(unittest.TestCase):
    def test_dsp_is_released_only_after_stream_closes(self):
        order = []
        stream = Mock()
        stream.abort.side_effect = lambda: order.append('abort')
        stream.close.side_effect = lambda: order.append('stream close')
        result = close_audio_stream(stream, after_close=lambda: order.append('DSP close'))
        self.assertTrue(result['completed'])
        self.assertEqual(result['errors'], [])
        self.assertEqual(order, ['abort', 'stream close', 'DSP close'])

    def test_failed_abort_still_closes_and_remains_a_failure(self):
        stream = Mock()
        stream.abort.side_effect = RuntimeError('device disconnected')
        result = close_audio_stream(stream)
        stream.close.assert_called_once()
        self.assertTrue(result['completed'])
        self.assertIn('device disconnected', result['errors'][0])

    def test_hung_close_cannot_report_success_or_free_active_dsp(self):
        release, entered = threading.Event(), threading.Event()
        stream, dsp = Mock(), Mock()
        def block():
            entered.set()
            release.wait()
        stream.close.side_effect = block
        try:
            result = close_audio_stream(stream, after_close=dsp, timeout=.02)
            self.assertTrue(entered.is_set())
            self.assertFalse(result['completed'])
            self.assertIn('exceeded', result['errors'][-1])
            dsp.assert_not_called()
        finally:
            release.set()
