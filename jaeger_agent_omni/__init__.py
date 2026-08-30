"""Jaeger Agent Omni — a native multimodal, agentic runtime.

The package keeps its extension surfaces visible at the top level:

``nodes``
    Neural capabilities that exchange learned representations with cognition.
``tools``
    Symbolic actions the cognition model may invoke.
``skills``
    Procedures that coordinate tools and nodes.
``memory``
    Persistent agent state.
``core``
    Model plumbing, streaming runtimes, and the imported Jaeger agent loop.

Audio reaches Gemma as learned representations, not as an ASR transcript.
Text is an output and a trusted tool-control format, not the input bridge.
Production speech currently streams decoded response text through the resident
Kokoro neural synthesizer; the tensor-native talker remains experimental.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .core import neural
from .core.capture import CaptureReader, CaptureWriter, Turn
from .core.neural import (
    JaegerAudioProjector,
    ModalityProjector,
    TurnEndDetector,
    TurnEndHead,
)
from .core.outputs import CognitiveOutput, ToolCall
from .core.performance_spec import DELIVERIES, EMOTIONS, PerformanceSpec, neutral
from .core.streaming import EncoderCache, StreamingChunker

__all__ = [
    "CaptureReader",
    "CaptureWriter",
    "CognitiveOutput",
    "DELIVERIES",
    "EMOTIONS",
    "EncoderCache",
    "JaegerAudioProjector",
    "ModalityProjector",
    "PerformanceSpec",
    "StreamingChunker",
    "ToolCall",
    "Turn",
    "TurnEndDetector",
    "TurnEndHead",
    "neural",
    "neutral",
    "__version__",
]
