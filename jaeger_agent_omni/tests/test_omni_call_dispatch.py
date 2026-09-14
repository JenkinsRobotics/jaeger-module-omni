"""Gemma's bare ``call:name{args}`` form must dispatch when it lands on its
own line after prose — the frozen a:ms_calc_and_save failure was a leaked,
never-executed ``call:write_file`` line."""
import unittest

from jaeger_agent_omni.core.agentic.omni_runtime import OmniSessionAdapter


class _FakeSession:
    """Minimal stand-in: parse_response never touches the session."""
    agentic_buffering = False

    def respond(self, *_a, **_kw):
        return ""

    def continue_agentic(self, *_a, **_kw):
        return None

    def cancel(self):
        pass


class BareCallDispatch(unittest.TestCase):
    def setUp(self):
        self.adapter = OmniSessionAdapter(_FakeSession())

    def test_bare_call_after_prose_is_dispatched(self):
        """Frozen a:ms_calc_and_save failure: Gemma answers, then puts the
        call on its own line. The write must actually be dispatched."""
        message = self.adapter.parse_response(
            'The sum is 1807. Saving it now.\n'
            'call:write_file{"path": "sums.txt", "content": "1807"}'
        )

        self.assertEqual(message["tool_calls"][0]["name"], "write_file")
        self.assertEqual(
            message["tool_calls"][0]["arguments"],
            {"path": "sums.txt", "content": "1807"},
        )
        self.assertNotIn("call:", message["content"] or "")

    def test_multiple_bare_call_lines_are_all_dispatched(self):
        message = self.adapter.parse_response(
            'call:write_file{"path": "a.txt", "content": "x"}\n'
            'call:append_file{"path": "a.txt", "content": "y"}'
        )

        self.assertEqual(
            [call["name"] for call in message["tool_calls"]],
            ["write_file", "append_file"],
        )
        self.assertIsNone(message["content"])

    def test_call_word_inside_prose_is_still_not_dispatched(self):
        message = self.adapter.parse_response(
            'Please call:calculate{expression:"10-3"} if needed.'
        )

        self.assertNotIn("tool_calls", message)


if __name__ == "__main__":
    unittest.main()
