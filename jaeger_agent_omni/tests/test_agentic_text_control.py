from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from research.benchmark_agentic_corpus import native_text_control, artifact_snapshot


class AgenticTextControlTests(unittest.TestCase):
    def test_text_uses_same_turn_envelope_without_audio(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [ord(c) for c in text]
        class Session:
            tokenizer=Tokenizer()
            def open_turn(self): pass
            def _prompt_parts(self): return [1,2],[3,4]
            def _prefill(self, ids): self.ids=ids
            def respond(self,**kw): return 'ok'
            def feed_audio(self,*args): raise AssertionError('text control encoded audio')
        session=Session()
        self.assertEqual(native_text_control(session,{'eos':[9]},'hello',max_tokens=32),'ok')
        self.assertEqual(session.ids,[1,2,*map(ord,'hello'),3])
        self.assertEqual(session._response_seed,4)
        self.assertTrue(session._turn_closed)

    def test_artifacts_read_actual_file_content(self):
        with TemporaryDirectory() as directory:
            p=Path(directory)/'skills/math.txt';p.parent.mkdir();p.write_text('1093')
            self.assertEqual(artifact_snapshot(directory)['skills/math.txt']['content'],'1093')
            p.unlink()
            self.assertEqual(artifact_snapshot(directory),{})
