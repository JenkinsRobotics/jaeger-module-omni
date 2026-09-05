"""CPU-only checks for tier routing and mounting unchanged packaged weights."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from jaeger_agent_omni import converse


class ConverseEarTests(unittest.TestCase):
    def test_tier_loads_only_its_existing_node(self):
        for mode, node, entry in (
            ("half", "jaegeromni-41m-medium", "build_converse"),
            ("quasi", "jaegeromni-41m-medium", "build_converse"),
            ("full", "jaegeromni-41m-converse", "build"),
        ):
            module = Mock()
            spec = Mock()
            with self.subTest(mode=mode), patch.object(
                converse.importlib.util, "spec_from_file_location", return_value=spec
            ) as locate, patch.object(
                converse.importlib.util, "module_from_spec", return_value=module
            ), patch.dict(sys.modules):
                self.assertIs(converse.load_trained_semantic(mode),
                              getattr(module, entry).return_value)
                locate.assert_called_once_with(
                    "nn_semantic", converse.HERE / "nodes" / node / "runtime.py")
                getattr(module, entry).assert_called_once_with()
                getattr(module, "build" if entry == "build_converse"
                        else "build_converse").assert_not_called()
        with self.assertRaises(KeyError):
            converse.load_trained_semantic("unknown")

    def test_medium_mount_preserves_pairing_and_transcript_prompt(self):
        spec = importlib.util.spec_from_file_location(
            "medium_test", converse.HERE / "nodes/jaegeromni-41m-medium/runtime.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        core = Mock()
        ear = core.CleanroomTranscriber.return_value
        ear.tokenizer.apply_chat_template.return_value = "before<AUDIO>after"
        ear.tokenizer.encode.side_effect = [[1], [2]]
        with patch.dict(sys.modules, {"gemmaomni_adapter": core,
                                     "mlx_audio_streaming": Mock(AUDIO_MARKER="<AUDIO>")}), \
             patch.object(sys, "path", sys.path.copy()):
            self.assertIs(module.build_converse(), ear)
        kwargs = core.CleanroomTranscriber.call_args.kwargs
        self.assertEqual(kwargs["adapter_path"], module.NODE_ROOT / "weights/adapter.npz")
        self.assertEqual(kwargs["encoder"]["source"],
                         str(module.NODE_ROOT / "weights/encoder"))
        messages = ear.tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(messages[0]["content"],
                         "Repeat the spoken words exactly. Output only what was said, "
                         "with no commentary or explanation.")
        self.assertEqual((ear.prefix_ids, ear.suffix_ids), ([1], [2]))

    def test_context_checkpoint_metadata_never_enters_parameter_tree(self):
        with patch.object(sys, "path", sys.path.copy()):
            sys.path.insert(0, str(converse.HERE / "core"))
            import gemmaomni_adapter as ca
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "adapter.npz"
            weights = np.arange(6, dtype=np.float32).reshape(2, 3)
            np.savez(checkpoint, arch=np.frombuffer(b"context_v2", dtype=np.uint8),
                     hidden=np.array([2]), lineage=np.array("qwen_free"),
                     **{"up.weight": weights})
            original = checkpoint.read_bytes()
            with patch.object(ca, "build_context_adapter") as build:
                adapter, arch = ca.load_any_adapter(checkpoint)
            self.assertIs(adapter, build.return_value)
            self.assertEqual(arch, "context_v2")
            self.assertEqual(build.call_args.args, (2,))
            params = build.call_args.kwargs["arrays"]
            self.assertEqual(set(params), {"up.weight"})
            np.testing.assert_array_equal(params["up.weight"], weights)
            self.assertEqual(checkpoint.read_bytes(), original)

    def test_bare_converse_selects_quasi_and_explicit_modes_survive(self):
        for argv, expected in ((["--converse"], "quasi"),
                               (["--converse", "half"], "half"),
                               (["--converse", "full"], "full")):
            with patch.object(sys, "argv", ["converse", *argv]), \
                 patch.object(converse, "converse", return_value=0) as run:
                self.assertEqual(converse.main(), 0)
                run.assert_called_once_with(expected, "stop", False, "nn", "nn")

    def test_wer_scores_heard_from_the_selected_tier_with_real_scorer(self):
        import soundfile as sf

        repo = converse.HERE.parent
        spec = importlib.util.spec_from_file_location(
            "wer_test", repo / "research/converse_ear_wer.py")
        harness = importlib.util.module_from_spec(spec)
        with patch.object(sys, "path", sys.path.copy()):
            spec.loader.exec_module(harness)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            audio = tmp / "clip.wav"
            sf.write(audio, np.zeros(16000, np.float32), 16000)
            manifest = tmp / "manifest.jsonl"
            manifest.write_text(json.dumps({"id": "fixture", "audio": str(audio),
                                            "text": "hello world"}) + "\n")
            out = tmp / "wer.json"
            ear = Mock()
            ear.respond.return_value = ("HEARD: hello there\nREPLY: world", {"total_s": 0.1})
            argv = ["converse_ear_wer.py", "--tier", "half", "--heard-reply",
                    "--manifest", str(manifest), "--output", str(out)]
            with patch.object(sys, "argv", argv), \
                 patch.object(sys, "path", [str(repo / "training"), *sys.path]), \
                 patch.object(converse, "load_trained_semantic", return_value=ear) as load, \
                 patch.object(harness, "sha256", return_value="fixture"):
                self.assertEqual(harness.main(), 0)
            load.assert_called_once_with("half")
            ear.transcribe.assert_not_called()
            self.assertEqual(ear.respond.call_args.kwargs,
                             {"system": converse.NN_SYSTEM, "history": [], "max_tokens": 140})
            result = json.loads(out.read_text())
            self.assertEqual(result["node"], "jaegeromni-41m-medium")
            self.assertEqual((result["edits"], result["reference_words"]), (1, 2))
            self.assertEqual(result["wer"], 0.5)
            self.assertEqual(result["rows"][0]["hypothesis"], "hello there")
