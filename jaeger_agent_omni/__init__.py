"""Jaeger Agent Omni — native multimodal neural agent for JaegerOS.

Parallel development to `jaeger-agent`, not a replacement for it:

    jaeger-agent          sequential pipeline, text as the transfer format
    jaeger-agent-omni     neural module integration, tensors as the transfer format

Audio reaches cognition as learned representations rather than as a transcript,
so prosody, hesitation and speaker state survive the trip. Text remains a
parallel output — it is no longer the only road.

    jaeger_agent_omni/    the runtime
      neural/             modality projectors and heads
      capture.py          turn log format, with a transcript-confidence gate
      streaming.py        1 s chunking + rolling encoder cache
      performance_spec.py voice + face intent
      outputs.py          text / speech / tool-call routing
    research/             experiment harness — stubs, wiring, inspection
    training/             corpus policy, losses, run configuration

Architecture is ported from MiniCPM-o 4.5, read out of its weights and
llama.cpp implementation. See `dev/minicpm-o measured architecture.md`.

Status: research. `node.py` is a stub and nothing is discoverable yet —
`dev/docs/reality/STATUS.md` is the honest account.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import neural
from .capture import CaptureReader, CaptureWriter, Turn
from .neural import JaegerAudioProjector, ModalityProjector, TurnEndDetector, TurnEndHead
from .outputs import CognitiveOutput, ToolCall
from .performance_spec import DELIVERIES, EMOTIONS, PerformanceSpec, neutral
from .streaming import EncoderCache, StreamingChunker

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
