"""StepAudio2: Audio tokenizer and TTS model package."""

from .token2wav import Token2wav
from .stepaudio2 import StepAudio2Base, StepAudio2

# Export classes from flashcosyvoice for backward compatibility
from .flashcosyvoice.modules.flow import CausalMaskedDiffWithXvec as CausalMaskedDiffWithXvecFlash

# Export classes from cosyvoice2 (used in flow.yaml configs)
from .cosyvoice2.flow.flow import CausalMaskedDiffWithXvec
from .cosyvoice2.transformer.upsample_encoder_v2 import UpsampleConformerEncoderV2
from .cosyvoice2.flow.flow_matching import CausalConditionalCFM
from .cosyvoice2.flow.decoder_dit import DiT

# Export utility classes that might be referenced in configs
from .cosyvoice2.transformer.attention import RelPositionMultiHeadedAttention
from .cosyvoice2.transformer.embedding import EspnetRelPositionalEncoding
from .cosyvoice2.transformer.subsampling import LinearNoSubsampling
from .cosyvoice2.transformer.encoder_layer import ConformerEncoderLayer
from .cosyvoice2.transformer.positionwise_feed_forward import PositionwiseFeedForward

# Export HiFTGenerator if needed
from .flashcosyvoice.modules.hifigan import HiFTGenerator

__all__ = [
    'Token2wav',
    'StepAudio2Base',
    'StepAudio2',
    'CausalMaskedDiffWithXvec',
    'CausalMaskedDiffWithXvecFlash',
    'UpsampleConformerEncoderV2',
    'CausalConditionalCFM',
    'DiT',
    'RelPositionMultiHeadedAttention',
    'EspnetRelPositionalEncoding',
    'LinearNoSubsampling',
    'ConformerEncoderLayer',
    'PositionwiseFeedForward',
    'HiFTGenerator',
]
