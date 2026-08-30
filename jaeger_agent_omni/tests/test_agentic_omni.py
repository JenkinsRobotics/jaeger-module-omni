import threading
import unittest

from pydantic import BaseModel

from jaeger_agent_omni.core.agentic.loop.jaeger_agent import JaegerAgent
from jaeger_agent_omni.core.agentic.omni_runtime import (
    NATIVE_MULTIMODAL_TURN,
    OMNI_BLOCKED_TOOLS,
    OmniSessionAdapter,
    omni_tools,
    prepare_agentic_system,
)
from jaeger_agent_omni.core.agentic._vendor.jaeger_os.core.tools.tool_schema import (
    ToolDef,
)


class _EchoArgs(BaseModel):
    value: str


class _FakeSession:
    def __init__(self, replies):
        self.replies = list(replies)
        self.controls = []
        self.cancelled = False

    def continue_agentic(self, value):
        self.controls.append(value)
        return len(value)

    def respond(self, max_tokens=0, eos_ids=(), on_token=None):
        reply = self.replies.pop(0)
        if on_token:
            on_token(7, 1)
        return reply

    def cancel(self):
        self.cancelled = True


class AgenticOmniTests(unittest.TestCase):
    def test_native_audio_turn_dispatches_tool_without_transcript_handoff(self):
        observed = []

        def echo(value):
            observed.append(value)
            return {"echo": value}

        tool = ToolDef(
            name="unit_echo", description="Echo a value.",
            args_model=_EchoArgs, fn=echo, side_effect="read")
        session = _FakeSession([
            '<|tool_call>call:unit_echo{"value":"hello"}<tool_call|>',
            "The echo is hello.",
        ])
        adapter = OmniSessionAdapter(
            session, eos_ids={1, 106}, max_tokens=32,
            initial_tools=[tool])
        agent = JaegerAgent(
            adapter, tools=[tool], max_iterations=3,
            blocked_tools=OMNI_BLOCKED_TOOLS)

        answer = agent.run_turn(NATIVE_MULTIMODAL_TURN)

        self.assertEqual(answer, "The echo is hello.")
        self.assertEqual(observed, ["hello"])
        self.assertEqual(len(session.controls), 1)
        self.assertIn('"tool_result"', session.controls[0])
        self.assertNotIn(NATIVE_MULTIMODAL_TURN, session.controls[0])

    def test_interrupt_reaches_native_session_at_token_boundary(self):
        session = _FakeSession(["stopped"])
        adapter = OmniSessionAdapter(session)
        formatted = adapter.format_messages(
            [{"role": "user", "content": NATIVE_MULTIMODAL_TURN}], [], "")
        interrupted = threading.Event()
        interrupted.set()
        adapter.call(formatted, interrupted)
        self.assertTrue(session.cancelled)

    def test_omni_catalog_excludes_cascade_listen_and_parallel_tts(self):
        preparation = prepare_agentic_system("Base voice policy.")
        names = {tool.name for tool in preparation.initial_tools}
        self.assertNotIn("listen", names)
        self.assertNotIn("text_to_speech", names)
        self.assertIn("load_tools", names)
        self.assertIn("ACTIVE TOOL SCHEMAS", preparation.system)
        self.assertEqual(names, {tool.name for tool in omni_tools()})

    def test_selected_session_must_support_agentic_continuation(self):
        class Incomplete:
            def respond(self):
                return ""

        with self.assertRaisesRegex(TypeError, "cannot append"):
            OmniSessionAdapter(Incomplete())

    def test_runtime_interrupt_reaches_agent_loop_and_native_generator(self):
        events = []

        class Agent:
            def interrupt(self):
                events.append("agent")

        class Runtime:
            agent = Agent()
            _native_cancel = staticmethod(lambda: events.append("native"))

        from jaeger_agent_omni.core.agentic.omni_runtime import OmniAgentRuntime

        OmniAgentRuntime.interrupt(Runtime())
        self.assertEqual(events, ["agent", "native"])


if __name__ == "__main__":
    unittest.main()
