import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from jaeger_agent_omni.benchmark import _physical_payload, build_gates
from jaeger_agent_omni.core.benchmarking import (
    GateResult,
    GateSpec,
    read_json,
    report_payload,
    run_gate,
    summarize,
    write_report,
)
from research.benchmark_agentic_corpus import (
    CaseDeadline, CorpusPermissionProvider, _expected_tools, _score, _selected,
    case_deadline,
)


class BenchmarkHarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_command_output_and_failure_are_durable(self):
        passed = run_gate(
            GateSpec("pass", "cpu", "passes",
                     (sys.executable, "-c", "print('measured')")),
            cwd=self.root, log_dir=self.root / "logs")
        failed = run_gate(
            GateSpec("fail", "cpu", "fails",
                     (sys.executable, "-c", "raise SystemExit(7)")),
            cwd=self.root, log_dir=self.root / "logs")

        self.assertEqual(passed.status, "PASS")
        self.assertIn("measured", Path(passed.log).read_text())
        self.assertEqual(failed.status, "FAIL")
        self.assertEqual(failed.returncode, 7)

    def test_required_skip_makes_report_incomplete_not_ready(self):
        rows = [
            GateResult("ok", "cpu", "ok", "PASS", True),
            GateResult("room", "physical", "room", "SKIP", True),
        ]
        summary = summarize(rows)
        self.assertFalse(summary["complete"])
        self.assertFalse(summary["profile_pass"])
        self.assertEqual(summary["required_skips"], ["room"])

    def test_optional_skip_does_not_block_release(self):
        rows = [
            GateResult("ok", "cpu", "ok", "PASS", True),
            GateResult("reference", "evidence", "ref", "SKIP", False),
        ]
        self.assertTrue(summarize(rows)["profile_pass"])

    def test_incremental_report_is_json_and_markdown(self):
        report = report_payload(
            profile="cpu", root=self.root,
            results=[GateResult("ok", "cpu", "works", "PASS", True)],
            git_commit="a" * 40, dirty=False)
        write_report(report, self.root / "report")

        loaded = read_json(self.root / "report/report.json")
        self.assertTrue(loaded["summary"]["profile_pass"])
        self.assertFalse(loaded["summary"]["release_ready"])
        self.assertIn("Verdict: PROFILE PASS",
                      (self.root / "report/report.md").read_text())

    def test_release_profile_declares_physical_evidence_required(self):
        args = SimpleNamespace(
            profile="release", front_end="jaegeromni-41m-medium",
            model="/tmp/model", physical_report=None)
        gates = build_gates(args, self.root / "run")
        physical = next(gate for gate in gates if gate.id == "physical_five_minute")
        self.assertTrue(physical.required)
        self.assertIn("5-minute", physical.skip_reason)
        self.assertIn("aec_room_stress", {gate.id for gate in gates})
        self.assertIn("agentic_cases12", {gate.id for gate in gates})
        self.assertIn("agentic_full_corpus", {gate.id for gate in gates})

    def test_omni_corpus_adapts_legacy_speech_tool_to_output_router(self):
        from jaeger_agent_omni.core.agentic.bench import CASES

        case = next(item for item in CASES if item.id == "speak_file")
        expected, adaptations = _expected_tools(case)
        self.assertEqual(expected, ["read_file"])
        self.assertIn("automatic_output_router_replaces_text_to_speech",
                      adaptations)

    def test_corpus_scoring_rejects_markup_and_forbidden_calls(self):
        from jaeger_agent_omni.core.agentic.bench import BenchCase

        case = BenchCase(
            id="unsafe", prompt="do not", forbidden_tools=["delete_file"])
        row = _score(
            case, ["delete_file"], [], "<|tool_call>bad", error=None)
        self.assertFalse(row["clean_output"])
        self.assertFalse(row["safety_ok"])
        self.assertFalse(row["case_pass"])

    def test_neural_vad_resolves_repository_owned_gate(self):
        from jaeger_agent_omni.core.runtime.neural_vad import GATE_ROOT

        self.assertEqual(GATE_ROOT.name, "silero-vad")
        self.assertTrue((GATE_ROOT / "gate.json").is_file())

    def test_agent_benchmark_has_no_legacy_runtime_dependency(self):
        import inspect
        from jaeger_agent_omni.tools import bench

        self.assertNotIn("jaeger_ai", inspect.getsource(bench))

    def test_agent_corpus_case_deadline_is_bounded(self):
        started = time.monotonic()
        with self.assertRaises(CaseDeadline):
            with case_deadline(0.02):
                time.sleep(1)
        self.assertLess(time.monotonic() - started, 0.5)

    def test_agent_corpus_limit_keeps_multiturn_session_whole(self):
        selected = _selected("a", [], ["mt_calc_reuse_1"], 1)
        self.assertEqual(
            [case.id for _lane, case in selected],
            ["mt_calc_reuse_1", "mt_calc_reuse_2"],
        )

    def test_agent_corpus_only_auto_approves_local_writes(self):
        from jaeger_agent_omni.core.agentic._vendor.jaeger_os.core.safety.permissions import (
            PermissionRequest, PermissionTier,
        )

        provider = CorpusPermissionProvider()
        request = lambda tier: PermissionRequest(tier, "bench", "probe")
        self.assertTrue(provider.confirm(request(PermissionTier.WRITE_LOCAL)))
        self.assertFalse(provider.confirm(request(PermissionTier.EXTERNAL_EFFECT)))
        self.assertFalse(provider.confirm(request(PermissionTier.HARDWARE)))

    def test_physical_gate_requires_every_release_interaction(self):
        payload = {
            "phase": "physical-room", "errors": [],
            "audio_shutdown": {"completed": True, "errors": []},
            "configuration": {"duplex_mode": "full"},
            "metrics": {
                "fed_seconds": 299.5, "dropped_seconds": 0,
                "mic_queue_overflow_blocks": 0, "turns": 3,
                "completed_replies": 3, "barge_events": 1,
                "speech_delivery": {"completed_uninterrupted_replies": 1},
                "barge_stops": 1, "interrupts": 1, "context_rolls": 1,
                "vad": {"mode": "neural"},
                "end_to_audio": {"n": 3, "median_ms": 600},
                "aec": {"available": True, "seconds_cancelled": 4},
                "companion": {"enabled": True, "respond": 2, "listen": 1},
                "agentic": {"enabled": True, "tool_calls": 3,
                    "tool_results": [{"name": "calculate", "tool_call_id": str(i),
                                      "content": '{"result": 4}'} for i in range(3)]},
            },
        }
        self.assertTrue(_physical_payload(payload)[0])
        payload["metrics"]["agentic"]["tool_calls"] = 2
        ok, detail, metrics = _physical_payload(payload)
        self.assertFalse(ok)
        self.assertIn("agent tools exercised", detail)
        self.assertFalse(metrics["checks"]["agent tools exercised"])

    def test_failed_tools_and_interrupted_audio_do_not_make_a_physical_pass(self):
        from jaeger_agent_omni.benchmark import successful_physical_tool_results
        failed = {'tool_calls': 3, 'tool_results': [
            {'name': 'load_tools', 'content': '{"ok": true}'},
            {'name': 'calculate', 'content': '{"error": "invalid"}'},
            {'name': 'write_file', 'content': '{"written": false}'}]}
        self.assertEqual(successful_physical_tool_results(failed), [])
        duplicate = {'name': 'calculate', 'tool_call_id': 'one', 'content': '{"result": 4}'}
        self.assertEqual(len(successful_physical_tool_results({'tool_results': [duplicate]*3})), 1)
        payload = {'errors': [], 'metrics': {'completed_replies': 3,
            'speech_delivery': {'completed_uninterrupted_replies': 0}, 'agentic': failed}}
        checks = _physical_payload(payload)[2]['checks']
        self.assertFalse(checks['successful action tool results'])
        self.assertFalse(checks['uninterrupted spoken reply'])
        self.assertFalse(checks['audio shutdown completed'])


if __name__ == "__main__":
    unittest.main()
