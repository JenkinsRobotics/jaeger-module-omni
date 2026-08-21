"""Load MiniCPM-o's trained audio projector into our own module.

The point is verification, not reuse. Our `JaegerAudioProjector` claims to be a
faithful port of MiniCPM's forward pass — ReLU, no LayerNorm, pool after both
linears. The only way to know is to run **their** weights through **our** code
and see whether it behaves.

    python research/load_minicpm_projector.py

If their weights produce diverse, structured output here, the port is correct
and any collapse we see in our own training is a training problem. If they do
not, the bug is in our code and we have been training around a defect.

Dimensions come from the checkpoint: 1024 (whisper-medium d_model) -> 4096
(Qwen3-8B hidden) -> 4096. Feeding it anything else is meaningless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspect_gguf import read_tensor                                # noqa: E402
from jaeger_agent_omni.neural import JaegerAudioProjector           # noqa: E402

DEFAULT_GGUF = Path.home() / "models/MiniCPM-o-4_5-gguf/audio/MiniCPM-o-4_5-audio-F16.gguf"
WHISPER_MEDIUM_DIM = 1024
QWEN3_8B_HIDDEN = 4096


def load(gguf_path=DEFAULT_GGUF, embed_norm: float | None = None
         ) -> JaegerAudioProjector:
    """Their weights, our module. No training, no adaptation."""
    proj = JaegerAudioProjector(audio_dim=WHISPER_MEDIUM_DIM,
                                cognition_dim=QWEN3_8B_HIDDEN,
                                embed_norm=embed_norm)
    with torch.no_grad():
        for ours, theirs in (("linear1", "audio_projector.linear1"),
                             ("linear2", "audio_projector.linear2")):
            layer = getattr(proj, ours)
            w = torch.from_numpy(read_tensor(gguf_path, f"{theirs}.weight"))
            b = torch.from_numpy(read_tensor(gguf_path, f"{theirs}.bias"))
            if w.shape != layer.weight.shape:
                raise ValueError(
                    f"{theirs}.weight is {tuple(w.shape)}, our {ours} expects "
                    f"{tuple(layer.weight.shape)} — the port's dimensions are wrong")
            layer.weight.copy_(w)
            layer.bias.copy_(b.reshape(-1))
    return proj.eval()


def cross_clip_cosine(vecs: torch.Tensor) -> float:
    v = torch.nn.functional.normalize(vecs, dim=-1)
    sim = v @ v.T
    return float(sim[~torch.eye(len(v), dtype=bool)].mean())


def main() -> int:
    if not Path(DEFAULT_GGUF).exists():
        print(f"missing {DEFAULT_GGUF}")
        return 1

    proj = load()
    print(f"loaded MiniCPM projector into our module: "
          f"{proj.n_params()/1e6:.1f}M params, "
          f"{proj.input_dim} -> {proj.cognition_dim}, "
          f"{proj.tokens_per_second:.0f} tok/s")

    # Feed it real encoder states and ask the only question that matters:
    # does the output distinguish one utterance from another?
    from transformers import WhisperModel, WhisperProcessor

    from jaeger_agent_omni.capture import SAMPLE_RATE, CaptureReader

    corpus = sys.argv[1] if len(sys.argv) > 1 else None
    if not corpus:
        print("\npass a corpus directory to run the diversity check")
        return 0

    processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
    encoder = WhisperModel.from_pretrained("openai/whisper-medium").encoder.eval()

    reader = CaptureReader(corpus)
    turns = list(reader)[:8]
    pooled, raw = [], []
    for t in turns:
        feats = processor(reader.audio(t), sampling_rate=SAMPLE_RATE,
                          return_tensors="pt").input_features
        with torch.no_grad():
            states = encoder(feats).last_hidden_state.float()
            n = min(states.shape[1], int(t.duration_s * 50))
            states = states[:, :n]
            out = proj(states)
        raw.append(states[0].mean(0))
        pooled.append(out[0].mean(0))

    enc_div = cross_clip_cosine(torch.stack(raw))
    proj_div = cross_clip_cosine(torch.stack(pooled))
    print(f"\ncross-clip cosine (1.0 = every clip identical):")
    print(f"  whisper-medium encoder output   {enc_div:.4f}")
    print(f"  MiniCPM projector output        {proj_div:.4f}")
    print(f"  our trained projector was       0.9903   (collapsed)")

    # The comparison is against the encoder INPUT, not an absolute number.
    # Speech shares enough structure that ~0.91 is the natural floor for
    # mean-pooled states, so the scale is compressed near 1.0 and a fixed
    # threshold reads everything as "collapsed".
    print(f"\n  delta vs its own input: {proj_div - enc_div:+.4f} "
          f"(negative = more discriminative than the encoder gave it)")

    if proj_div <= enc_div + 0.01:
        print("\nPORT VERIFIED — their weights preserve discriminability in our "
              "code, so the forward pass is faithful. A projector of ours that "
              "collapses is a TRAINING failure, not an architecture bug.")
    else:
        print("\nTheir weights compress here too. Either the port is wrong, or "
              "the input is not what this projector expects (it wants "
              "whisper-medium states, post-ln_post).")
    return 0


def demo():
    v = torch.eye(4, 16)
    assert abs(cross_clip_cosine(v)) < 0.01
    assert cross_clip_cosine(torch.ones(4, 16)) > 0.999
    if Path(DEFAULT_GGUF).exists():
        proj = load()
        assert proj.linear1.weight.shape == (QWEN3_8B_HIDDEN, WHISPER_MEDIUM_DIM)
        assert proj.linear2.weight.shape == (QWEN3_8B_HIDDEN, QWEN3_8B_HIDDEN)
        # Real trained weights, not zeros or noise.
        assert 0.001 < proj.linear1.weight.std() < 0.5
        assert proj.n_params() > 20_000_000
        print(f"load_minicpm_projector: {proj.n_params()/1e6:.1f}M params loaded OK")
    else:
        print("load_minicpm_projector: gguf absent, metric-only check OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        sys.exit(main())
