import unittest
from unittest.mock import patch
from pydantic import BaseModel
from jaeger_agent_omni.core.agentic.omni_runtime import AgenticPreparation, OmniAgentRuntime
from jaeger_agent_omni.core.agentic.loop.jaeger_agent import JaegerAgent
from jaeger_agent_omni.core.agentic._vendor.jaeger_os.core.tools.tool_schema import ToolDef
from jaeger_agent_omni.core.runtime.companion import parse_decision

class Value(BaseModel):
    value: str

class Session:
    def __init__(self):
        self.controls = []
        self.replies = iter([
            '<|tool_call>call:calculate{"value":"6*7"}<tool_call|>',
            '<|tool_call>call:write_file{"value":"42"}<tool_call|>',
            'DECISION: RESPOND\nRESPONSE: Saved forty-two to the file.',
        ])
    def respond(self, **kwargs):
        return next(self.replies)
    def continue_agentic(self, value):
        self.controls.append(value)
    def cancel(self):
        pass

class NativeContinuationTests(unittest.TestCase):
    def test_native_turn_finishes_actions_and_emits_companion_decision(self):
        calls = []
        def calculate(value):
            calls.append(('calculate', value))
            return {'result': 42}
        def write_file(value):
            calls.append(('write_file', value))
            return {'written': True}
        tools = [ToolDef(name=name, description=name, args_model=Value,
                         fn=fn, side_effect='read')
                 for name, fn in [('calculate', calculate), ('write_file', write_file)]]
        session = Session()
        with patch('jaeger_agent_omni.core.agentic.workspace.ensure_bound'):
            runtime = OmniAgentRuntime(session, AgenticPreparation('', tuple(tools)),
                                       load_skills=False)
        # Keep the runtime's real shortcut policy, but dispatch only harmless
        # deterministic in-memory tools for this control-flow regression.
        runtime.agent = JaegerAgent(runtime.adapter, tools=tools,
                                    skip_final_tools=runtime.agent.skip_final_tools)
        answer = runtime.respond_native()
        self.assertEqual(calls, [('calculate', '6*7'), ('write_file', '42')])
        self.assertEqual(len(session.controls), 2)
        self.assertTrue(parse_decision(answer).should_respond)

if __name__ == '__main__':
    unittest.main()
