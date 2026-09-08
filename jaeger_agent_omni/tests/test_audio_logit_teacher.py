import unittest
import numpy as np


class AudioTeacherTest(unittest.TestCase):
    def test_teacher_uses_audio_and_cannot_receive_gradients(self):
        import mlx.core as mx
        from training.cleanroom_context_ce import audio_teacher_logits, OUTPUT_DIM
        previous = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            class TinyBackbone:
                def target_logits(self, ids, embeddings, first, count):
                    # A controlled model that exposes the actual conditioning.
                    return mx.sum(embeddings, axis=1)
            tokens = mx.ones((1, 6, 4))
            audio = mx.full((2, 4), 3 * np.sqrt(OUTPUT_DIM))
            def teacher(x):
                return mx.sum(audio_teacher_logits(TinyBackbone(), None, tokens,
                                                    x, 2, 2, 4, 2))
            self.assertAlmostEqual(float(teacher(audio)), 40.0, places=4)
            self.assertTrue(np.all(np.asarray(mx.grad(teacher)(audio)) == 0))
        finally:
            mx.set_default_device(previous)
