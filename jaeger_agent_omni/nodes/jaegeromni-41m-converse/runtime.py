"""Entry point for the consolidated converse semantic channel: the
jaegeromni-41m-causal-balanced adapter serving the converse seam.

Consolidation step 2 (2026-09-03). Pairs the SAME frozen stock
whisper-medium encoder (byte-identical to the causal-balanced node's) with
a metadata-cleaned copy of the causal-balanced streaming_causal_v1 adapter,
wrapped in the adapter's training-time transcript prompt — measured 5.03%
devclean64 at this seam versus 5.65% for the imported gemmaomni-v2 ear
(prompt alignment recovers the own-seam number exactly; see
research/candidate_evals/causal-balanced-at-converse-seam-sysprompt-*).

The checkpoint carries trained overlap/speaker_change/turn_end/speaker_id
heads but NO trained directed head, so build() drops "directed" from
perceive() output: converse's admission gate then defaults to 1.0 (never
pre-blocks) and the post-generation text-wake decision stays authoritative.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NODE_ROOT = Path(__file__).resolve().parent
BACKBONE = NODE_ROOT.parent.parent / "core"   # gemmaomni core modules

# The adapter was trained and gated with this system prompt in context
# (training/stock_whisper_gemma.py TRANSCRIPT_SYSTEM). Measured 2026-09-03:
# without it this checkpoint pays +1.04 points devclean64 at this seam.
TRANSCRIPT_SYSTEM = (
    "Repeat the spoken words exactly. Output only what was said, with no "
    "commentary or explanation."
)


def manifest() -> dict:
    return json.loads((NODE_ROOT / "node.json").read_text())


def build(*, model_path=None, system=None, device="auto"):
    """Construct the live transcriber for this immutable pairing."""
    sys.path.insert(0, str(BACKBONE))
    import gemmaomni_adapter as ca
    import mlx_audio_streaming as mas

    spec = manifest()["encoder"]
    local = NODE_ROOT / spec["path"]
    transcriber = ca.CleanroomTranscriber(
        adapter_path=NODE_ROOT / manifest()["adapter"]["path"],
        model_path=model_path,
        encoder={"source": str(local) if local.is_dir() else spec["id"],
                 "output_dim": spec["output_dim"]},
    )
    rendered = transcriber.tokenizer.apply_chat_template(
        [{"role": "system", "content": system or TRANSCRIPT_SYSTEM},
         {"role": "user", "content": mas.AUDIO_MARKER}],
        tokenize=False, add_generation_prompt=True)
    prefix, suffix = rendered.split(mas.AUDIO_MARKER, 1)
    transcriber.prefix_ids = transcriber.tokenizer.encode(
        prefix, add_special_tokens=False)
    transcriber.suffix_ids = transcriber.tokenizer.encode(
        suffix, add_special_tokens=False)

    # No trained directed head in this checkpoint: report no directed
    # evidence instead of a random logit, so the admission pre-filter
    # fails open to the post-generation text-wake decision.
    original_perceive = transcriber.adapter.perceive

    def perceive(values, self_speech=None):
        embeddings, heads = original_perceive(values, self_speech)
        heads.pop("directed", None)
        return embeddings, heads

    transcriber.adapter.perceive = perceive
    return transcriber
