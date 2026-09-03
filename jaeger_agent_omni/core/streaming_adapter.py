#!/usr/bin/env python3
"""CAUSAL context adapter + perception heads — streaming, barge-in, speaker ID.

The keypoint adapter (3.23% WER) uses BIDIRECTIONAL attention over a whole
sealed utterance. That is fine for batch transcription and fatal for real-time:

  * it cannot emit anything until the turn ends -> no streaming
  * every frame's representation depends on future frames -> a speaker-change
    or overlap signal at time t would be "predicted" using audio from t+5s,
    which is unusable live and dishonest offline

One change fixes both: mask the attention so frame t sees only frames <= t.
Then the same forward pass can (a) emit embeddings incrementally as audio
arrives and (b) carry per-frame heads whose outputs are legitimate at the
moment they fire.

Heads (per pooled frame, 25 Hz — 40 ms resolution):

    speaker_change   p(a different voice starts here)   -> speaker ID / diarization
    overlap          p(more than one voice active)      -> BARGE-IN signal
    turn_end         p(the utterance is completing)     -> endpointing without VAD

These are the perception outputs the sealed towers structurally cannot provide,
and they are why the owned seam exists. Behaviour stays a POLICY over them:
"yield on direct address, keep talking through backchannels" is a decision made
with these signals, not a reflex hardcoded in a VAD.

Weights load from a bidirectional keypoint (identical parameter shapes); only
the mask differs, so a short fine-tune adapts it rather than training anew.

    python streaming_adapter.py --selftest      # CPU, no models needed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ENC_DIM, OUT_EMB, POOL = 1024, 2560, 2
HEADS = ("speaker_change", "overlap", "turn_end", "directed")

# ADDRESSEE DETECTION — "was that said to me, or to someone else?"
#
# A wake phrase answers this reliably and unnaturally: it makes you address a
# machine every turn. For a companion sitting in a shared room the question is
# not "did they say the magic word" but "was this turn aimed at me", and that
# is a genuinely learnable signal — people speak to devices differently from
# how they speak to each other. Device speech is louder, more carefully
# articulated, more canonical in syntax and far less disfluent.
#
# The head lives on the SAME trunk as the perception heads, so it costs one
# extra 2048->1 matrix on a forward pass that already happened. It also sees
# what a transcript cannot: prosody and articulation survive in the encoder
# states and are gone by the time the words are text.
#
# It does NOT replace the wake phrase. The phrase stays as a guaranteed way in
# when the head is unsure; the head is what makes the common case natural.

# Per-frame speaker IDENTITY, not merely "something changed".
#   0 nobody   1 first talker   2 second talker   3 both at once
# speaker_change stalled at F1 37.4% because a binary boundary label is a
# shadow of the real target: it says a change occurred and refuses to say
# between whom. The synthetic mixes knew the identity exactly all along and
# threw it away. With identity, "change" is derivable (argmax transitions)
# rather than learned, which is strictly more information for the same audio.
SPEAKER_CLASSES = 4

# SELF-VOICE AWARENESS.
# The agent hears its own speech through the mic. Without knowing which frames
# are its own output, the overlap head fires on every reply and the agent
# reads itself as a constant interruption. We DO know our playback waveform,
# so the frame-aligned answer to "was I speaking here?" is free information —
# it is supplied as an explicit input rather than left for the model to guess.
# 0.0 = mic only, 1.0 = agent was playing audio during this frame.


def build_streaming_adapter(hidden: int = 2048, seed: int = 0, arrays=None,
                            causal: bool = True, enc_dim: int = ENC_DIM):
    """Causal context adapter with three per-frame perception heads.

    `enc_dim` is the encoder boundary width and comes from the node manifest
    (1024 for whisper-medium, 1280 for large-v3). Checkpoints carry their own
    shapes, so it is inferred from `arrays` when one is supplied — a loaded
    checkpoint is the authority on the width it was trained at.
    """
    if arrays is not None and "ln.weight" in arrays:
        enc_dim = int(arrays["ln.weight"].shape[0])
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten

    class StreamingAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(enc_dim)
            self.up = nn.Linear(enc_dim, hidden)
            self.attn = nn.MultiHeadAttention(hidden, 8)
            self.ln_a = nn.LayerNorm(hidden)
            self.ff1 = nn.Linear(hidden, hidden * 2)
            self.ff2 = nn.Linear(hidden * 2, hidden)
            self.ln_f = nn.LayerNorm(hidden)
            self.out = nn.Linear(hidden, OUT_EMB)
            # perception heads: tiny (2048->1 each), cannot disturb the
            # transcription path when their gradients are disabled
            self.head_speaker_change = nn.Linear(hidden, 1)
            self.head_overlap = nn.Linear(hidden, 1)
            self.head_turn_end = nn.Linear(hidden, 1)
            self.head_speaker_id = nn.Linear(hidden, SPEAKER_CLASSES)
            self.head_directed = nn.Linear(hidden, 1)
            # projects the per-frame self-speech flag into the trunk width so
            # attention can attend to "this stretch is me, not them"
            self.self_proj = nn.Linear(1, hidden)

        def _trunk(self, values, self_speech=None):
            h = nn.gelu(self.up(self.ln(values)))
            usable = (h.shape[0] // POOL) * POOL
            if usable == 0:
                h = mx.mean(h, axis=0, keepdims=True)
            else:
                h = mx.mean(h[:usable].reshape(-1, POOL, hidden), axis=1)
            if self_speech is not None:
                # additive, zero-init projection: with no flag supplied (or an
                # all-zero flag) the trunk is bit-identical to before, so
                # existing checkpoints and WER are unaffected.
                f = mx.array(np.asarray(self_speech, dtype=np.float32)[:h.shape[0]])
                if f.shape[0] < h.shape[0]:
                    f = mx.concatenate([f, mx.zeros(h.shape[0] - f.shape[0])])
                h = h + self.self_proj(f[:, None])
            x = h[None, :]
            n = x.shape[1]
            mask = None
            if causal and n > 1:
                # frame t attends to <= t only. Additive -inf mask above the
                # diagonal; this is what makes live emission honest.
                mask = mx.where(
                    mx.tril(mx.ones((n, n))) > 0, 0.0, -1e9).astype(x.dtype)
            x = self.ln_a(x + self.attn(x, x, x, mask=mask))
            x = self.ln_f(x + self.ff2(nn.gelu(self.ff1(x))))
            return x

        def __call__(self, values, self_speech=None):
            return self.out(self._trunk(values, self_speech)[0])

        def perceive(self, values, self_speech=None):
            """Embeddings + per-frame perception logits, one forward pass.

            self_speech: optional (frames,) array, 1.0 where the agent was
            playing audio. Supplying it lets the overlap head distinguish
            "someone interrupted me" from "that is my own voice".
            """
            x = self._trunk(values, self_speech)
            return self.out(x[0]), {
                "speaker_change": self.head_speaker_change(x)[0, :, 0],
                "overlap": self.head_overlap(x)[0, :, 0],
                "turn_end": self.head_turn_end(x)[0, :, 0],
                "directed": self.head_directed(x)[0, :, 0],
                # (frames, 4) logits — argmax gives who is talking
                "speaker_id": self.head_speaker_id(x)[0],
            }

    mx.random.seed(seed)
    adapter = StreamingAdapter()
    # zero-init so an untrained/absent self-flag is a strict no-op
    adapter.self_proj.weight = mx.zeros_like(adapter.self_proj.weight)
    adapter.self_proj.bias = mx.zeros_like(adapter.self_proj.bias)
    if arrays is not None:
        flat = [(k, mx.array(v)) for k, v in arrays.items()
                if k not in ("arch", "hidden", "causal")]
        adapter.update(tree_unflatten(flat))
    mx.eval(adapter.parameters())
    return adapter


def directed_score(logits) -> float:
    """One utterance-level score from the per-frame directed logits.

    Mean over frames, not max: a single confident frame is exactly what a
    stray word from the television produces, while genuine address holds the
    property across the whole turn. Averaging makes the head argue its case
    for the duration rather than win on one spike.
    """
    v = np.asarray(logits, dtype=np.float32)
    return float(v.mean()) if v.size else 0.0


def change_from_identity(speaker_id_logits) -> "np.ndarray":
    """Derive speaker-change from identity — do not learn it separately.

    A frame is a change when the predicted talker differs from the previous
    frame's. Deriving it means the change signal inherits the identity head's
    accuracy instead of fighting a 2%-positive binary target on its own.
    """
    ids = np.asarray(speaker_id_logits).argmax(axis=-1)
    change = np.zeros(len(ids), dtype=np.int8)
    change[1:] = (ids[1:] != ids[:-1]).astype(np.int8)
    return change


def load_keypoint_into_streaming(path: Path, hidden: int = 2048):
    """Warm-start from the bidirectional keypoint.

    Shapes are identical — only the attention mask differs — so the trunk
    transfers directly and only needs adapting, not relearning. The three
    heads are absent from that checkpoint and start fresh.
    """
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    hidden = int(arrays.get("hidden", [hidden])[0])
    ad = build_streaming_adapter(hidden, arrays=arrays)
    missing = [h for h in HEADS if not any(f"head_{h}" in k for k in arrays)]
    return ad, hidden, missing


def save_streaming(path, adapter, hidden: int) -> None:
    from mlx.utils import tree_flatten
    out = {k: np.array(v) for k, v in tree_flatten(adapter.parameters())}
    out["arch"] = np.frombuffer(b"streaming_causal_v1", dtype=np.uint8)
    out["hidden"] = np.array([hidden], dtype=np.int64)
    out["causal"] = np.array([1], dtype=np.int64)
    np.savez(path, **out)


def _selftest() -> int:
    """CPU-only: proves causality, streaming equivalence, and head wiring."""
    import mlx.core as mx
    mx.set_default_device(mx.cpu)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, ENC_DIM)).astype(np.float32)
    ad = build_streaming_adapter(256, seed=0)          # small for speed

    full, heads = ad.perceive(mx.array(x))
    n = full.shape[0]
    assert full.shape == (n, OUT_EMB), full.shape
    for h in HEADS:
        assert heads[h].shape == (n,), (h, heads[h].shape)
    assert heads["speaker_id"].shape == (n, SPEAKER_CLASSES), heads["speaker_id"].shape
    print(f"shapes ok: {x.shape} -> emb {full.shape}, heads {n} frames each")

    # CAUSALITY: appending future audio must not change earlier frames.
    prefix = ad(mx.array(x[:100]))
    m = prefix.shape[0]
    drift = float(mx.max(mx.abs(full[:m] - prefix)))
    print(f"causality drift over first {m} frames: {drift:.2e}")
    assert drift < 1e-4, "future audio changed past frames — not causal"

    # the bidirectional variant MUST fail that same check (control)
    bi = build_streaming_adapter(256, seed=0, causal=False)
    bi_full = bi(mx.array(x))
    bi_pre = bi(mx.array(x[:100]))
    bi_drift = float(mx.max(mx.abs(bi_full[:bi_pre.shape[0]] - bi_pre)))
    print(f"bidirectional control drift: {bi_drift:.2e} (expected large)")
    assert bi_drift > 1e-3, "control failed: bidirectional should drift"

    # STREAMING: emit in chunks, compare to one-shot
    chunks, acc = [], []
    for i in range(0, 200, 40):
        acc.append(x[i:i + 40])
        chunks.append(ad(mx.array(np.concatenate(acc))))
    inc = chunks[-1]
    assert inc.shape == full.shape
    print(f"streaming: {len(chunks)} incremental emissions, final matches "
          f"one-shot within {float(mx.max(mx.abs(inc - full))):.2e}")

    # heads are separable: zeroing them must not touch embeddings
    import mlx.nn as nn
    ad.head_overlap.weight = mx.zeros_like(ad.head_overlap.weight)
    emb2, h2 = ad.perceive(mx.array(x))
    assert float(mx.max(mx.abs(emb2 - full))) == 0.0
    assert float(mx.max(mx.abs(h2["overlap"] - ad.head_overlap.bias))) < 1e-5
    print("heads are independent of the transcription path")

    # identity -> change must be exact, since it is derived not learned
    fake = np.zeros((6, SPEAKER_CLASSES), np.float32)
    for i, who in enumerate([1, 1, 2, 2, 2, 0]):
        fake[i, who] = 5.0
    got = change_from_identity(fake).tolist()
    assert got == [0, 0, 1, 0, 0, 1], got
    print(f"speaker-change derived from identity: {got} (transitions only)")

    # SELF-VOICE FLAG: two properties that must both hold.
    # (a) zero-init => supplying it cannot change transcription, so adding it
    #     to a validated checkpoint is safe by construction, not by luck;
    # (b) once self_proj is non-zero the flag MUST move the output, otherwise
    #     the input is silently disconnected and training it does nothing.
    ad2 = build_streaming_adapter(256, seed=0)
    base = ad2(mx.array(x))
    nf = base.shape[0]
    for flag, what in ((np.zeros(nf), "all-zero"), (np.ones(nf), "all-one")):
        d = float(mx.max(mx.abs(ad2(mx.array(x), self_speech=flag) - base)))
        assert d == 0.0, f"self_speech {what} perturbed a zero-init trunk: {d}"
    ad2.self_proj.weight = mx.ones_like(ad2.self_proj.weight) * 0.01
    moved = float(mx.max(mx.abs(ad2(mx.array(x), self_speech=np.ones(nf)) - base)))
    assert moved > 1e-4, "self_speech disconnected — training it would be a no-op"
    print(f"self-voice flag: no-op at init, live once trained ({moved:.1e})")
    print("\nSELFTEST PASS — causal, streamable, heads wired")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--from-keypoint", default=None)
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if args.from_keypoint:
        ad, hid, missing = load_keypoint_into_streaming(Path(args.from_keypoint))
        print(f"loaded keypoint, hidden={hid}")
        print(f"heads needing training: {missing}")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
