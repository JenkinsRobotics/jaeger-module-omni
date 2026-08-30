"""End-to-end wiring: capture -> encoder -> projector -> backbone -> loss.

Runs the whole Stage 3/4 path on stub models, so the plumbing is proven before
11 GB of weights and a rented GPU are involved. When the real run breaks, this
tells you whether it is the pipeline or the models.

The one thing it actually proves: **the projector can learn.** The overfit check
below drives a real training loop on a handful of samples and asserts the loss
falls. If wiring is wrong — detached graph, frozen parameters, shape mismatch
that broadcasts silently — that assert fails.

    python research/pipeline.py     # self-check, no model download
"""

import math

import numpy as np
import torch
import torch.nn as nn

from jaeger_agent_omni.core.capture import CaptureReader, CaptureWriter, SAMPLE_RATE
from jaeger_agent_omni.core.neural import JaegerAudioProjector
from distill import DistillWeights, agreement_rate, total_loss
from stubs import StubAudioEncoder, StubBackbone


class OmniPath(nn.Module):
    """audio -> encoder(frozen) -> projector(trained) -> backbone(frozen)

    Everything except the projector is frozen. That is the Stage 3 configuration
    and the reason this is affordable: gradients still flow *through* the
    backbone, but no optimiser state is kept for it.
    """

    def __init__(self, encoder, backbone, projector):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()
        self.backbone = backbone.requires_grad_(False).eval()
        self.projector = projector

    def forward(self, waveform: torch.Tensor, output_hidden_states: bool = False):
        audio = self.encoder(waveform).last_hidden_state
        embeds = self.projector(audio)
        return self.backbone(inputs_embeds=embeds,
                             output_hidden_states=output_hidden_states)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def text_teacher(backbone, token_ids: torch.Tensor):
    """The other half of self-distillation: same weights, text input."""
    with torch.no_grad():
        return backbone(input_ids=token_ids, output_hidden_states=True)


def demo():
    torch.manual_seed(0)
    import tempfile

    audio_dim, cognition_dim, vocab = 64, 96, 128

    encoder = StubAudioEncoder(audio_dim=audio_dim)
    backbone = StubBackbone(hidden_size=cognition_dim, vocab_size=vocab, n_layers=2)
    projector = JaegerAudioProjector(audio_dim=audio_dim, cognition_dim=cognition_dim)
    path = OmniPath(encoder, backbone, projector)

    # Only the projector trains. If this drifts, training cost explodes silently.
    trainable = path.trainable_parameters()
    assert len(trainable) == len(list(projector.parameters())), \
        "something other than the projector is trainable"
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in path.parameters())
    assert n_train < n_total * 0.6, (n_train, n_total)

    # --- capture round-trip feeds the pipeline -----------------------------
    with tempfile.TemporaryDirectory() as tmp:
        w = CaptureWriter(tmp)
        for i in range(4):
            tone = np.sin(np.arange(SAMPLE_RATE) * (0.02 + 0.01 * i)).astype(np.float32)
            w.write(tone, transcript=f"question {i}", response=f"answer {i}", soul="mochi")

        reader = CaptureReader(tmp)
        turns = list(reader)
        assert len(turns) == 4
        waveform = torch.tensor(np.stack([reader.audio(t) for t in turns]))
        assert waveform.shape == (4, SAMPLE_RATE)

    # 1 s of audio -> 50 encoder frames -> 10 audio tokens.
    out = path(waveform, output_hidden_states=True)
    assert out.logits.shape == (4, 10, vocab), out.logits.shape

    # --- self-distillation shapes -----------------------------------------
    token_ids = torch.randint(0, vocab, (4, 7))
    teacher = text_teacher(backbone, token_ids)
    assert teacher.logits.shape == (4, 7, vocab)

    loss, parts = total_loss(
        DistillWeights.distill(),
        student_hidden=out.hidden_states, teacher_hidden=teacher.hidden_states,
        student_logits=out.logits, teacher_logits=teacher.logits,
        labels=torch.randint(0, vocab, (4, 10)))
    assert set(parts) == {"hidden", "kl", "ce"}
    assert 0.0 <= agreement_rate(out.logits, teacher.logits) <= 1.0

    # --- the check that matters: can the projector actually learn? ---------
    # Overfit four fixed samples. If wiring is broken, loss will not move.
    torch.manual_seed(1)
    labels = torch.randint(0, vocab, (4, 10))
    opt = torch.optim.AdamW(path.trainable_parameters(), lr=3e-3)
    warm = DistillWeights.warmup()

    first = last = None
    for step in range(250):
        opt.zero_grad()
        out = path(waveform)
        loss, _ = total_loss(warm, student_logits=out.logits, labels=labels)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    # Below the uniform-prior floor ln(vocab) means it genuinely memorised the
    # targets rather than merely sharpening toward the marginal distribution.
    # That is the property worth asserting; a bare ratio would pass on noise.
    uniform_floor = math.log(vocab)
    assert first > uniform_floor * 0.95, f"suspiciously low start: {first:.3f}"
    assert last < uniform_floor, \
        f"projector never beat the uniform floor: {last:.3f} >= {uniform_floor:.3f}"
    assert last < first * 0.75, f"projector barely learned: {first:.3f} -> {last:.3f}"

    # Frozen really means frozen — no gradient reached the backbone.
    assert all(p.grad is None for p in backbone.parameters()), \
        "backbone accumulated gradients despite requires_grad_(False)"

    # Gradient still flowed THROUGH the backbone to the projector. This is the
    # thing people assume is free and is not — it is why VRAM stays high.
    assert projector.linear1.weight.grad.abs().sum() > 0

    print(f"pipeline: {n_train/1e3:.0f}K trainable of {n_total/1e3:.0f}K total "
          f"({100*n_train/n_total:.0f}%)")
    print(f"pipeline: overfit loss {first:.3f} -> {last:.3f} in 250 steps "
          f"(uniform floor {uniform_floor:.3f})")
    print("pipeline: capture->encoder->projector->backbone->loss->step OK")


if __name__ == "__main__":
    demo()
