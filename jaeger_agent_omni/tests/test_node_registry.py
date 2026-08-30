import unittest
from pathlib import Path

from jaeger_agent_omni.node_registry import (
    discover,
    load_factory,
    select_profile,
    validate_profiles,
)
from jaeger_agent_omni.gate_registry import discover_gates, select_gate_profile


class NodeRegistryTests(unittest.TestCase):
    def test_real_registry_is_flat_complete_and_loadable(self):
        root = Path(__file__).resolve().parents[1] / "nodes"
        packages = discover(root)
        self.assertEqual(len(packages), 14)
        self.assertIn("gemmaomni-41m-medium-final4000", packages)
        self.assertIn("jaegeromni-41m-causal-balanced", packages)
        self.assertIn("jaegeromni-41m-audio-perception", packages)
        self.assertIn("kokoro-tts", packages)
        self.assertIn("gemma-minicpm-hidden-talker", packages)
        self.assertIn("whisper-large-v3-turbo-stt", packages)
        self.assertIn("neural-acoustic-controller", packages)
        for package in packages.values():
            self.assertTrue(callable(load_factory(package)))
        profiles = validate_profiles(root, packages)
        self.assertEqual(profiles["default"], "jaegeromni-41m-medium")
        selected = select_profile(root)
        self.assertIn("kokoro-tts", selected)
        self.assertNotIn("stock-whisper-large-v3-turbo50m", selected)
        shadow = select_profile(root, "neural-acoustic-controller-shadow")
        self.assertIn("neural-acoustic-controller", shadow)
        self.assertIn("jaegeromni-41m-medium", shadow)

    def test_gates_are_separate_from_neural_nodes(self):
        root = Path(__file__).resolve().parents[2]
        nodes = discover(root / "jaeger_agent_omni" / "nodes")
        gates = discover_gates(root / "gates")
        self.assertEqual(set(gates), {"silero-vad", "speex-aec"})
        self.assertTrue(set(nodes).isdisjoint(gates))
        for package in gates.values():
            self.assertTrue(callable(load_factory(package)))
        selected = select_gate_profile(root / "gates")
        self.assertEqual(set(selected), set(gates))


if __name__ == "__main__":
    unittest.main()
