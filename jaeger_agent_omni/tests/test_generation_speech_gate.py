import unittest
import numpy as np
from jaeger_agent_omni.core.runtime.neural_vad import SpeechPresence


class GenerationSpeechGateTests(unittest.TestCase):
    def test_noise_energy_cannot_cancel_and_speech_requires_confirmation(self):
        probabilities = iter([0.0, 0.9, 0.9, 0.0, 0.9, 0.9, 0.9, 0.0])
        gate = SpeechPresence(lambda frame: next(probabilities))
        loud = np.full(480, 0.08, dtype=np.float32)
        self.assertGreater(float(np.sqrt(np.mean(loud**2))), 0.012 * 1.1)
        self.assertFalse(gate.update(loud[:240]))
        self.assertEqual(gate.frames, 0)
        self.assertFalse(gate.update(loud[240:]))
        self.assertEqual([gate.update(loud) for _ in range(7)],
                         [False, False, False, False, False, True, False])
