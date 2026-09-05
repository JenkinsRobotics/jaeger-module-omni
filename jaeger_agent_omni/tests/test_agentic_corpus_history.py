import unittest
from unittest.mock import Mock

import numpy as np

from research.benchmark_agentic_corpus import native_turn, reset_case_session


class AgenticCorpusHistoryTests(unittest.TestCase):
    def test_new_case_cannot_inherit_discovered_tool_visibility(self):
        from jaeger_agent_omni.core.skill_registry import toolset_scoping
        toolset_scoping.enable_toolset("files")
        self.assertIn("files", toolset_scoping.active_toolset_names())
        session = Mock()
        reset_case_session(session)
        self.assertNotIn("files", toolset_scoping.active_toolset_names())
        session.reset.assert_called_once_with()

    def test_native_siblings_keep_cache_and_feed_only_audio(self):
        session = Mock()
        session.respond.return_value = "answer"
        first, second = np.ones(2000), np.zeros(1800)
        for audio in (first, second):
            self.assertEqual(native_turn(session, {"eos": [106]}, audio,
                                         max_tokens=96), "answer")
        session.reset.assert_not_called()
        session.feed_text.assert_not_called()
        self.assertEqual(session.open_turn.call_count, 2)
        self.assertEqual(session.end_of_turn.call_count, 2)
        chunks = [c.args[0] for c in session.feed_audio.call_args_list]
        np.testing.assert_array_equal(np.concatenate(chunks), np.concatenate([first, second]))
        session.respond.assert_called_with(max_tokens=96, eos_ids=[106])
