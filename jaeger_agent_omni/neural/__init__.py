"""Jaeger neural modules — specialists that talk to cognition through projectors.

    audio    → whisper encoder → JaegerAudioProjector  ─┐
    vision   → SigLIP          → JaegerVisionProjector ─┼→ shared cognition
    memory   → retrieval net   → JaegerMemoryProjector ─┘

Every modality follows the same pattern: trim padding, compress time, project
into cognition's embedding space. `base.ModalityProjector` owns that arithmetic
so each new specialist is a known job rather than a research question.

Heads go the other way — cognition state out to a decision or a control signal.
"""

from .audio import JaegerAudioProjector
from .base import ModalityProjector, REGISTRY, get, pooled_tokens, register, hidden_size_of, embedding_norm_of
from .delta import DeltaAdapter, MiniCPMDeltaPath, SpeechOutDelta
from .gate import AudioGate, embed
from .heads import TurnEndDetector, TurnEndHead, turn_end_loss
from .speech import GemmaTalkerConditioner, SpeechConditionerConfig

__all__ = [
    "embedding_norm_of",
    "hidden_size_of",
    "AudioGate",
    "DeltaAdapter",
    "MiniCPMDeltaPath",
    "SpeechOutDelta",
    "GemmaTalkerConditioner",
    "SpeechConditionerConfig",
    "JaegerAudioProjector",
    "ModalityProjector",
    "REGISTRY",
    "TurnEndDetector",
    "TurnEndHead",
    "embed",
    "get",
    "pooled_tokens",
    "register",
    "turn_end_loss",
]
