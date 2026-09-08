import unittest
from types import SimpleNamespace

import numpy as np

from research.joint_vision_probe import multimodal_turn


class JointVisionTest(unittest.TestCase):
    def session(self):
        calls = []
        class Session:
            mx = np
            model = object()
            cache = object()
            tokenizer = SimpleNamespace(encode=lambda text, **kw: [7, 8])
            def open_turn(self): calls.append(('open',))
            def _ensure_turn_prefix(self): calls.append(('prefix',))
            def _prefill(self, ids, embeddings=None): calls.append(('prefill', ids, embeddings))
            def feed_audio(self, samples): calls.append(('audio', np.array(samples)))
            def end_of_turn(self): calls.append(('end_audio',))
            def _prompt_parts(self): return [], [10, 11]
            def respond(self, **kw): return 'answer'
        return Session(), calls

    def test_raw_audio_and_visual_embeddings_enter_same_session(self):
        session, calls = self.session()
        model, cache = session.model, session.cache
        features = np.full((2, 2560), 3*np.sqrt(2560), np.float32)
        waveform = np.arange(1800, dtype=np.float32)
        answer = multimodal_turn(session, {'eos': [1]}, features=features,
                                 boundaries=[20, 21], audio=waveform)
        self.assertEqual(answer, 'answer')
        self.assertIs(session.model, model)
        self.assertIs(session.cache, cache)
        projected = next(c[2] for c in calls if c[0] == 'prefill' and c[2] is not None)
        np.testing.assert_allclose(projected, 3)
        np.testing.assert_array_equal(np.concatenate([c[1] for c in calls if c[0] == 'audio']), waveform)
        self.assertEqual(calls[-1], ('end_audio',))

    def test_malformed_visual_tokens_are_rejected(self):
        session, calls = self.session()
        with self.assertRaisesRegex(ValueError, 'width 2560'):
            multimodal_turn(session, {'eos': [1]}, features=np.ones((2, 8)), boundaries=[20, 21])
        self.assertEqual(calls, [])
