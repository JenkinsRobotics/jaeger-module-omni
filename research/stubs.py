"""Tiny stand-ins for Whisper and the cognition model.

The point is that the whole Stage 3/4 pipeline can be exercised end to end
without 11 GB of weights or a GPU. Every interface the real models expose and
we depend on is reproduced here, at toy dimensions, with random weights.

What we depend on and therefore must stub:

    encoder     .last_hidden_state at [B, T, audio_dim], 50 fps
    backbone    .get_input_embeddings(), forward(inputs_embeds=..., ...),
                output_hidden_states, .config.hidden_size

If a real run fails on an interface these stubs do not have, the stub is wrong
and should grow — that is the contract.

ponytail: hand-rolled 2-layer transformer instead of a hf-internal-testing tiny
model, so the checks run offline with no hub round-trip.

    python research/stubs.py     # self-check
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class StubConfig:
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int


@dataclass
class StubOutput:
    logits: torch.Tensor
    hidden_states: tuple = ()
    last_hidden_state: torch.Tensor = None


class StubAudioEncoder(nn.Module):
    """Stands in for `WhisperModel.encoder`. 50 frames per second of audio."""

    def __init__(self, audio_dim: int = 64, frame_rate: int = 50):
        super().__init__()
        self.audio_dim = audio_dim
        self.frame_rate = frame_rate
        self.proj = nn.Linear(1, audio_dim)
        self.block = nn.TransformerEncoderLayer(
            d_model=audio_dim, nhead=4, dim_feedforward=audio_dim * 2,
            batch_first=True, dropout=0.0)

    def forward(self, waveform: torch.Tensor) -> StubOutput:
        """[B, n_samples] at 16 kHz -> [B, n_frames, audio_dim]."""
        b, n = waveform.shape
        n_frames = max(1, int(round(n / 16000 * self.frame_rate)))
        # Crude framing: average-pool the waveform into frames. Enough to make
        # the output depend on the input, which is all a stub owes us.
        frames = torch.nn.functional.adaptive_avg_pool1d(
            waveform.unsqueeze(1), n_frames).transpose(1, 2)
        h = self.block(self.proj(frames))
        return StubOutput(logits=None, last_hidden_state=h)


class StubBackbone(nn.Module):
    """Stands in for a causal LM. Exposes only what the projector path uses."""

    def __init__(self, hidden_size: int = 96, vocab_size: int = 128, n_layers: int = 2):
        super().__init__()
        self.config = StubConfig(hidden_size, vocab_size, n_layers)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4,
                                       dim_feedforward=hidden_size * 2,
                                       batch_first=True, dropout=0.0)
            for _ in range(n_layers)])
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds=None, input_ids=None,
                output_hidden_states: bool = False, **_):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("need inputs_embeds or input_ids")
            inputs_embeds = self.embed(input_ids)

        h = inputs_embeds
        states = [h]
        causal = nn.Transformer.generate_square_subsequent_mask(h.shape[1], device=h.device)
        for layer in self.layers:
            h = layer(h, src_mask=causal)
            states.append(h)

        return StubOutput(logits=self.lm_head(h),
                          hidden_states=tuple(states) if output_hidden_states else (),
                          last_hidden_state=h)


def demo():
    torch.manual_seed(0)

    enc = StubAudioEncoder(audio_dim=64)
    # 2 s of audio at 16 kHz -> 100 frames, i.e. the real 50 fps rate.
    out = enc(torch.randn(2, 32000))
    assert out.last_hidden_state.shape == (2, 100, 64), out.last_hidden_state.shape
    assert enc(torch.randn(1, 16000)).last_hidden_state.shape == (1, 50, 64)

    # Output must actually depend on input, or every downstream check is vacuous.
    a = enc(torch.zeros(1, 16000)).last_hidden_state
    b = enc(torch.randn(1, 16000)).last_hidden_state
    assert not torch.allclose(a, b), "stub encoder ignores its input"

    lm = StubBackbone(hidden_size=96, vocab_size=128, n_layers=2)
    assert lm.config.hidden_size == 96

    # The two entry points the training loop uses.
    ids = torch.randint(0, 128, (2, 7))
    by_ids = lm(input_ids=ids, output_hidden_states=True)
    assert by_ids.logits.shape == (2, 7, 128)
    assert len(by_ids.hidden_states) == 3, "embeddings + one per layer"

    embeds = lm.get_input_embeddings()(ids)
    by_embeds = lm(inputs_embeds=embeds, output_hidden_states=True)
    # Both paths must agree — the audio path substitutes embeddings for ids and
    # relies on that equivalence.
    assert torch.allclose(by_ids.logits, by_embeds.logits, atol=1e-5)

    assert lm(inputs_embeds=embeds).hidden_states == (), "off by default"

    # Causal masking: a later token must not change earlier positions. If this
    # breaks, hidden-state distillation targets are contaminated by the future.
    x = lm.get_input_embeddings()(torch.randint(0, 128, (1, 6)))
    y = x.clone()
    y[:, -1] = torch.randn_like(y[:, -1])
    assert torch.allclose(lm(inputs_embeds=x).logits[:, :-1],
                          lm(inputs_embeds=y).logits[:, :-1], atol=1e-5), \
        "stub backbone is not causal"

    try:
        lm()
    except ValueError:
        pass
    else:
        raise AssertionError("must reject empty input")

    print(f"stubs: encoder {sum(p.numel() for p in enc.parameters())/1e3:.0f}K, "
          f"backbone {sum(p.numel() for p in lm.parameters())/1e3:.0f}K params")
    print("stubs: rates, both entry points, embed/id equivalence, causality OK")


if __name__ == "__main__":
    demo()
