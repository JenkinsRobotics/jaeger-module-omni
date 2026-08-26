import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from jaeger_agent_omni.model_nodes import (
    load_audio_node,
    validate_node_checkpoint,
)


class AudioNodeTests(unittest.TestCase):
    def test_audio_node_pins_adapter_and_boundary(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        adapter = root / "adapter.npz"
        adapter.write_bytes(b"owned adapter")
        encoder = root / "encoder"
        encoder.mkdir()
        (encoder / "model.safetensors").write_bytes(b"encoder")
        digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
        manifest = {
            "schema_version": 1,
            "kind": "jaeger_audio_node",
            "id": "test-node",
            "runtime": "stock_whisper_gemma",
            "status": "test",
            "encoder": {
                "id": "openai/whisper-medium",
                "output_dim": 1024,
                "boundary": "stock_whisper_encoder_50hz",
                "path": "encoder",
            },
            "adapter": {
                "path": "adapter.npz",
                "sha256": digest,
                "architecture": "arch-v1",
                "pool_step": 2,
            },
            "backbone": {"family": "gemma-4-e4b"},
        }
        (root / "node.json").write_text(json.dumps(manifest))
        node = load_audio_node(root)
        validate_node_checkpoint(node, {
            "encoder": "openai/whisper-medium",
            "input_dim": 1024,
            "input_boundary": "stock_whisper_encoder_50hz",
            "adapter_arch": "arch-v1",
            "pool_step": 2,
        })
        with self.assertRaisesRegex(ValueError, "input dimension"):
            validate_node_checkpoint(node, {
                "encoder": "openai/whisper-medium",
                "input_dim": 1280,
                "input_boundary": "stock_whisper_encoder_50hz",
                "adapter_arch": "arch-v1",
                "pool_step": 2,
            })

    def test_audio_node_rejects_modified_adapter(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        adapter = root / "adapter.npz"
        adapter.write_bytes(b"before")
        encoder = root / "encoder"
        encoder.mkdir()
        (encoder / "model.safetensors").write_bytes(b"encoder")
        manifest = {
            "schema_version": 1,
            "kind": "jaeger_audio_node",
            "id": "test-node",
            "runtime": "stock_whisper_gemma",
            "status": "test",
            "encoder": {
                "id": "x", "output_dim": 1, "boundary": "b",
                "path": "encoder",
            },
            "adapter": {
                "path": "adapter.npz",
                "sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
                "architecture": "a",
                "pool_step": 2,
            },
            "backbone": {"family": "gemma"},
        }
        (root / "node.json").write_text(json.dumps(manifest))
        adapter.write_bytes(b"after")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            load_audio_node(root)


if __name__ == "__main__":
    unittest.main()
