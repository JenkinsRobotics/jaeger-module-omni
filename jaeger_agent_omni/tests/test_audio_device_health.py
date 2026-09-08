"""Fault injection at the real duplex callback and lifecycle seams, CPU only."""
import importlib.util
from pathlib import Path
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'nodes/audio-duplex-io-v0/runtime.py'
spec = importlib.util.spec_from_file_location('audio_health_test_node', PATH)
node = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node)


class AudioHealthTest(unittest.TestCase):
    def device(self):
        with patch.object(node, 'pyaec') as aec, patch.object(node.sd, 'Stream') as stream:
            aec.Aec.return_value.cancel_echo.return_value = [0] * 160
            io = node.AudioIO()
        return io, stream.return_value

    def test_callback_failure_silences_and_surfaces(self):
        io, _ = self.device()
        io._aec.cancel_echo.side_effect = ValueError('injected DSP failure')
        output = np.ones((480, 1), dtype=np.int16)
        io._cb(output.copy(), output, 480, None, None)
        self.assertFalse(output.any())
        with self.assertRaisesRegex(RuntimeError, 'injected DSP failure'):
            io.check_health()
        with self.assertRaises(RuntimeError):
            io.play_chunk(np.ones(240))

    def test_missing_callbacks_surface(self):
        io, _ = self.device()
        io._last_callback = time.monotonic() - 3
        with self.assertRaisesRegex(RuntimeError, 'stopped delivering'):
            io.is_playing()

    def test_start_and_stop_failures_still_close(self):
        io, stream = self.device()
        stream.start.side_effect = RuntimeError('disconnected')
        with self.assertRaisesRegex(RuntimeError, 'disconnected'):
            io.__enter__()
        stream.close.assert_called_once()
        stream.close.reset_mock()
        stream.stop.side_effect = RuntimeError('stop failure')
        with self.assertRaisesRegex(RuntimeError, 'stop failure'):
            io.__exit__()
        stream.close.assert_called_once()

    def test_restart_uses_fresh_device_and_discards_stale_audio(self):
        io, old_stream = self.device()
        io._play_q.put(np.ones(480, dtype=np.int16))
        io.__exit__()
        old_stream.close.assert_called_once()
        new_io, new_stream = self.device()
        new_io.__enter__()
        self.assertTrue(new_io._play_q.empty())
        self.assertEqual(new_io.health['callbacks'], 0)
        new_stream.start.assert_called_once()
