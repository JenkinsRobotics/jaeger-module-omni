import unittest
from pathlib import Path
import numpy as np
from jaeger_agent_omni.node_registry import load_factory, load_package


class NeuralTurnNodeTest(unittest.TestCase):
    def test_quiet_speech_probability_opens_once_and_isolated_clicks_do_not(self):
        root = Path(__file__).resolve().parents[1]/'nodes/para-turn-state-neural-v1'
        factory = load_factory(load_package(root))
        cls = factory.__globals__['TurnStateNode']
        probabilities = iter([.9]*20 + [.01]*40)
        node = cls(probability=lambda _: next(probabilities))
        # Quiet waveform is deliberately below the old .02 RMS gate.
        frame = np.full(480, .001, np.float32)
        outputs = [node.process(frame, i*.03) for i in range(60)]
        self.assertTrue(outputs[3]['in_turn'])
        self.assertEqual(sum(r['turn_ended'] for r in outputs), 1)
        probabilities = iter(([.9] + [.01]*8)*5)
        node = cls(probability=lambda _: next(probabilities))
        self.assertFalse(any(node.process(frame, i*.03)['in_turn'] for i in range(45)))
