"""The duplex runtime — what Google did not ship.

Native Gemma 4 has a streaming-capable audio tower inside a turn-based chat
model. Everything that turns that into a conversation is here:

    gemma_audio.py   incremental prefill while the user is still speaking
    (turn-end)       neural/heads.py decides when it is our turn
    (arbitration)    outputs.py generation_id cancels stale work on barge-in
"""

from .gemma_audio import CHUNK_S, GemmaAudioPrefiller, TOKENS_PER_CHUNK
from .duplex import DuplexLoop, DuplexStats
from .session import StreamingSession, TurnTiming

__all__ = ["CHUNK_S", "DuplexLoop", "DuplexStats", "GemmaAudioPrefiller",
           "StreamingSession", "TOKENS_PER_CHUNK", "TurnTiming"]
