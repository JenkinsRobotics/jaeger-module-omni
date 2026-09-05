import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from jaeger_agent_omni.core.mlx_audio_streaming import AUDIO_MARKER, MlxGemmaAudioTranscriber


class ConverseContextPromptTests(unittest.TestCase):
    def test_context_belongs_only_to_current_audio_turn_and_prompt_state_restores(self):
        history = [{"role": "user", "content": "prior words"},
                   {"role": "assistant", "content": "prior answer"}]
        for context in (None, "the speaker is speaker S2"):
            tokenizer = Mock()
            tokenizer.apply_chat_template.return_value = "prefix"+AUDIO_MARKER+"suffix"
            tokenizer.encode.return_value = [10]
            ear = SimpleNamespace(tokenizer=tokenizer, prefix_ids=[1], suffix_ids=[2],
                                  max_tokens=3, transcribe=Mock(return_value=("hello", {})))
            raw, _ = MlxGemmaAudioTranscriber.respond(
                ear, np.zeros(16000), system="rules", history=history,
                max_tokens=140, user_context=context)
            messages = tokenizer.apply_chat_template.call_args.args[0]
            self.assertEqual(messages[0], {"role": "system", "content": "rules"})
            self.assertEqual(messages[1:3], history)
            self.assertEqual(messages[-1]["content"],
                             f"[context: {context}]\n{AUDIO_MARKER}" if context else AUDIO_MARKER)
            self.assertEqual((ear.prefix_ids, ear.suffix_ids, ear.max_tokens), ([1], [2], 3))
            self.assertEqual(raw, "HEARD:hello")
