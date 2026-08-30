import inspect
import unittest
from pathlib import Path

import numpy as np

from jaeger_agent_omni.node_registry import load_factory, load_package


ROOT = (Path(__file__).resolve().parents[1] / "nodes" /
        "neural-acoustic-controller")


class AcousticControllerTests(unittest.TestCase):
    def setUp(self):
        package = load_package(ROOT)
        self.factory = load_factory(package)

    def test_untrained_node_fails_closed_but_contract_can_be_tested(self):
        with self.assertRaisesRegex(RuntimeError, "no promoted checkpoint"):
            self.factory(device="cpu")
        controller = self.factory(device="cpu", initialize_untrained=True)
        self.assertGreater(controller.model.parameter_count, 200_000)
        self.assertLess(controller.model.parameter_count, 400_000)

    def test_shapes_passthrough_and_typed_five_stage_contract(self):
        controller = self.factory(device="cpu", initialize_untrained=True)
        rng = np.random.default_rng(7)
        mic = rng.normal(0, 0.01, 1600).astype(np.float32)
        reference = rng.normal(0, 0.01, 1600).astype(np.float32)
        output = controller.process(mic, reference)
        np.testing.assert_array_equal(output.cleaned_pcm, mic)
        self.assertEqual(output.embedding.shape, (10, 128))
        self.assertEqual(output.voice_probability.shape, (10,))
        self.assertEqual(set(output.scene_probabilities), {
            "silence", "environment", "self", "speech"})
        self.assertEqual(set(output.event_probabilities), {
            "external_speech", "self_speech", "music", "noise", "overlap"})
        self.assertEqual(set(output.turn_probabilities), {
            "forming", "committed"})
        for value in output.as_dict().values():
            if isinstance(value, np.ndarray):
                self.assertTrue(np.isfinite(value).all())

        # Drive deterministic probabilities through the exact typed chain.
        output.voice_probability[:] = 0.9
        for values in output.scene_probabilities.values():
            values[:] = 0.0
        output.scene_probabilities["speech"][:] = 0.9
        output.turn_probabilities["committed"][:] = 0.9
        output.wake_probability[:] = 0.8
        heard = controller.hear(output)
        categorised = controller.categorise(heard)
        turn = controller.state_of(
            categorised, semantic_commit_probability=0.9)
        context = controller.addressed_to_me(
            turn, semantic_address_probability=0.8)
        action = controller.decide(context)
        self.assertEqual(action.action, "answer")

    def test_network_is_causal_and_reset_is_deterministic(self):
        controller = self.factory(device="cpu", initialize_untrained=True)
        mic_a = np.zeros(320, dtype=np.float32)
        mic_b = mic_a.copy()
        mic_b[160:] = 0.8
        reference = np.zeros_like(mic_a)
        controller.reset()
        first = controller.process(mic_a, reference)
        controller.reset()
        second = controller.process(mic_b, reference)
        np.testing.assert_allclose(
            first.embedding[0], second.embedding[0], atol=1e-7)
        controller.reset()
        repeated = controller.process(mic_a, reference)
        np.testing.assert_allclose(first.embedding, repeated.embedding,
                                   atol=1e-7)

    def test_contract_has_no_text_boundary(self):
        controller = self.factory(device="cpu", initialize_untrained=True)
        for method in (
                controller.hear, controller.categorise, controller.state_of,
                controller.addressed_to_me, controller.decide):
            self.assertNotIn("text", inspect.signature(method).parameters)
        with self.assertRaisesRegex(ValueError, "multiple of 160"):
            controller.process(np.zeros(10), np.zeros(10))


if __name__ == "__main__":
    unittest.main()
