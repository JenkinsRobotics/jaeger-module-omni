"""Frozen Gemma vision tower/projector; never constructs a language model."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class FrozenVision:
    def __init__(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_vlm.models.gemma4.config import VisionConfig
        from mlx_vlm.models.gemma4.vision import VisionModel
        from mlx_vlm.models.gemma4.gemma4 import MultimodalEmbedder
        from transformers import Gemma4ImageProcessor
        from jaeger_agent_omni.node_registry import load_package, verify_weights

        verify_weights(load_package(ROOT))
        config = json.loads((ROOT / 'vision_config.json').read_text())
        vision = VisionConfig.from_dict(config['vision_config'])

        class Projection(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_tower = VisionModel(vision)
                self.embed_vision = MultimodalEmbedder(
                    vision.hidden_size, config['text_hidden_size'], vision.rms_norm_eps)

            def __call__(self, pixels, positions):
                return self.embed_vision(self.vision_tower(pixels, positions))

        self.network = Projection()
        self.network.load_weights(list(mx.load(str(ROOT / 'weights/vision.safetensors')).items()), strict=True)
        self.network.freeze()
        self.network.eval()
        mx.eval(self.network.parameters())
        self.processor = Gemma4ImageProcessor(**config['image_processor'])
        self.output_dim = config['text_hidden_size']
        self.image_start_id = config['boi_token_id']
        self.image_end_id = config['eoi_token_id']

    def encode(self, image):
        import mlx.core as mx
        import numpy as np
        batch = self.processor(images=[image], return_tensors='np', device='cpu')
        features = self.network(mx.array(batch['pixel_values']), mx.array(batch['image_position_ids']))
        mx.eval(features)
        expected = int(np.asarray(batch['num_soft_tokens_per_image']).sum())
        if features.shape != (1, expected, self.output_dim):
            raise ValueError(f'vision token contract mismatch: {features.shape}, expected {expected}')
        result = np.asarray(features.astype(mx.float32))[0].copy()
        if not np.isfinite(result).all():
            raise ValueError('nonfinite vision embeddings')
        return result


def build():
    return FrozenVision()
