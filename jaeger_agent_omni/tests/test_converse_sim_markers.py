import importlib.util
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class ConverseSimMarkerTests(unittest.TestCase):
    def test_assertion_echo_cannot_count_as_a_runtime_reply(self):
        spec = importlib.util.spec_from_file_location(
            "sim_marker_check", Path(__file__).with_name("converse_sim.py"))
        module = importlib.util.module_from_spec(spec)
        # The simulation installs a fake sounddevice; keep that isolated.
        with patch.dict(sys.modules):
            spec.loader.exec_module(module)
            tee = SimpleNamespace(cv=threading.Condition(), cursor=0,
                                  lines=["  [SIM] PASS image · ⏵ agent: blue circle"])
            with patch.object(module, "TEE", tee):
                self.assertIsNone(module.wait_for("⏵ agent:", timeout=0))
                self.assertFalse(module.seen_since("⏵ agent:", 0))
                tee.lines.append("  ⏵ agent: actual next reply")
                self.assertEqual(module.wait_for("⏵ agent:", timeout=0), tee.lines[1])
                self.assertTrue(module.seen_since("⏵ agent:", 0))
