"""Entrypoint for the JaegerOmni 41M Medium room/multidomain node."""

import json
import sys
from pathlib import Path

from jaeger_agent_omni.core.runtime.stock_whisper import StockWhisperGemmaSession

NODE_ROOT = Path(__file__).resolve().parent


def build(*, model_path, system=None, device="auto"):
    """Build this immutable encoder/adapter pairing."""
    return StockWhisperGemmaSession(
        NODE_ROOT, model_path=model_path, system=system, device=device)


def build_converse(*, model_path=None):
    """Expose the same immutable pairing at the endpointed converse seam.

    Bidirectional attention is preserved. Only completed turns use this ear;
    the full tier's forming preview continues to use the causal node.
    """
    sys.path.insert(0, str(NODE_ROOT.parent.parent / "core"))
    from gemmaomni_adapter import CleanroomTranscriber
    from mlx_audio_streaming import AUDIO_MARKER

    spec = json.loads((NODE_ROOT / "node.json").read_text())
    ear = CleanroomTranscriber(
        adapter_path=NODE_ROOT / spec["adapter"]["path"],
        model_path=model_path,
        encoder={"source": str(NODE_ROOT / spec["encoder"]["path"]),
                 "output_dim": spec["encoder"]["output_dim"]})
    # Match the checkpoint's training and locked WER protocol.
    rendered = ear.tokenizer.apply_chat_template(
        [{"role": "system", "content":
          "Repeat the spoken words exactly. Output only what was said, with no "
          "commentary or explanation."},
         {"role": "user", "content": AUDIO_MARKER}],
        tokenize=False, add_generation_prompt=True)
    prefix, suffix = rendered.split(AUDIO_MARKER, 1)
    ear.prefix_ids = ear.tokenizer.encode(prefix, add_special_tokens=False)
    ear.suffix_ids = ear.tokenizer.encode(suffix, add_special_tokens=False)
    return ear
