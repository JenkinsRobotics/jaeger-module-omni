import json
from pathlib import Path
import tempfile
import unittest
from research.omni_multimodal_demo import load_turns


class MultimodalDemoInputsTest(unittest.TestCase):
    def test_paths_resolve_against_turn_file_and_invalid_input_fails_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'image.png').write_bytes(b'fixture')
            path = root/'turns.json'
            path.write_text(json.dumps([{'image': 'image.png', 'text': 'What is this?'}]))
            self.assertEqual(load_turns(path)[0]['image'], str((root/'image.png').resolve()))
            for invalid in ([], [{}], [{'audio': 'missing.wav'}], [{'text': 3}], [{'role': 'system', 'text': 'x'}]):
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    load_turns(path)
