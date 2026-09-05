import base64
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from jaeger_agent_omni.converse import load_node


class ConverseVisionTests(unittest.TestCase):
    def test_local_image_reaches_declared_content_without_a_model(self):
        vision = load_node("semantic-vision-v0")
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "with spaces.png"
            Image.new("RGB", (10, 10), "blue").save(image)
            uri = vision.image_uri(image)
            prefix, encoded = uri.split(",", 1)
            self.assertEqual(prefix, "data:image/png;base64")
            self.assertEqual(base64.b64decode(encoded), image.read_bytes())
            content = vision.content("What color?", uri)
            self.assertEqual(content[0]["image_url"]["url"], uri)
            self.assertEqual(content[1]["text"], "What color?")
            with self.assertRaises(OSError):
                vision.image_uri(Path(tmp)/"missing.png")

    def test_roll_preserves_the_shared_audio_text_vision_history(self):
        brain = load_node("llm-gemma-v0")
        shared = []
        brain.history = shared
        for i in range(12):
            brain.remember(f"turn {i}", f"reply {i}")
        self.assertIs(brain.history, shared)
        self.assertEqual(len(shared), brain.MAX_MESSAGES)
        self.assertEqual(shared[-1]["content"], "reply 11")
        brain.clear()
        self.assertEqual(shared, [])
