"""Splice audio embeddings into the backbone's chat template.

from .neural.base import hidden_size_of

An instruct backbone expects its conversation markers. Handed bare embeddings
with no template, it has no idea it is in a dialogue and falls back to generic
instruction-following — observed directly:

    omni: "What is the sentiment of the following passage?"
    omni: "A: The first sentence is a good start, but..."

The text path never had this problem because `apply_chat_template` wraps the
transcript. The audio path must be wrapped the same way, with the projected
audio standing exactly where the user's words would have stood:

    [ system + <user> ][ AUDIO EMBEDS ][ </user> <assistant> ][ target ]
      prefix            projector out   suffix                  supervised

MiniCPM does the same thing, marking the span with `<audio_bos>`/`<audio_eos>`
inside the template rather than replacing it (mtmd.cpp:526).

Training and inference must use the identical wrapper. A model trained inside
the template and generated outside it sees a different distribution than the
one it learned, which is exactly the failure above.

    python -m jaeger_agent_omni.prompt     # self-check
"""

from __future__ import annotations

import torch

# Stands where the user's words go, so the template renders normally and the
# span can be located afterwards. Any token sequence works — it is replaced by
# embeddings, never read.
AUDIO_PLACEHOLDER = "\x00AUDIO\x00"

DEFAULT_SYSTEM = (
    "You are a helpful voice assistant. The user speaks to you directly. "
    "Answer in plain conversational English."
)

# Stage 3 warmup trains the model to REPEAT the utterance. Training that task
# under the assistant prompt teaches a contradiction — and it shows at
# inference as chat-refusal boilerplate ("I'm sorry for the confusion...")
# whenever audio grounding is weak. Measured on the first free-running delta
# eval: clips either transcribe nearly perfectly or refuse entirely. The
# system prompt must match the stage's task.
WARMUP_SYSTEM = (
    "Listen to the user and repeat exactly what they said, word for word. "
    "Output only their words."
)


class ChatWrapper:
    """Builds the token embeddings that bracket a span of audio.

    Computed once per (tokenizer, system prompt) and reused, because the
    template does not change between turns.
    """

    def __init__(self, tokenizer, backbone, *, system: str | None = DEFAULT_SYSTEM,
                 device=None, dtype=None):
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.system = system            # recorded into checkpoints/artifacts:
                                        # a train/eval prompt mismatch reads
                                        # as a worse model
        self.device = device or next(backbone.parameters()).device
        self.dtype = dtype or next(backbone.parameters()).dtype

        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": AUDIO_PLACEHOLDER})

        if getattr(tokenizer, "chat_template", None):
            rendered = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)
        else:
            # No template (base models, stubs): a minimal one still beats
            # nothing, because the model needs *some* signal that a turn is
            # starting and its own turn is expected.
            rendered = (f"{system}\n\n" if system else "")
            rendered += f"User: {AUDIO_PLACEHOLDER}\nAssistant:"

        if AUDIO_PLACEHOLDER not in rendered:
            raise ValueError(
                "chat template dropped the audio placeholder — cannot locate "
                "where audio belongs in the conversation")
        head, tail = rendered.split(AUDIO_PLACEHOLDER, 1)

        self.prefix_ids = self._encode(head)
        self.suffix_ids = self._encode(tail)

    def _encode(self, text: str) -> torch.Tensor:
        if not text:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        ids = self.tokenizer(text, return_tensors="pt",
                             add_special_tokens=False).input_ids[0]
        return ids.to(self.device)

    def _embed(self, ids: torch.Tensor, batch: int) -> torch.Tensor:
        if ids.numel() == 0:
            hidden = hidden_size_of(self.backbone)
            return torch.zeros(batch, 0, hidden, device=self.device, dtype=self.dtype)
        emb = self.backbone.get_input_embeddings()(ids.unsqueeze(0))
        return emb.expand(batch, -1, -1).to(self.dtype)

    @property
    def overhead_tokens(self) -> int:
        """Template cost per turn. Counts against the prefill budget."""
        return int(self.prefix_ids.numel() + self.suffix_ids.numel())

    def wrap(self, audio_embeds: torch.Tensor) -> tuple:
        """[B, A, D] -> ([B, P+A+S, D], n_prefix_and_audio).

        The second value is where the supervised region begins: everything
        before it is conversation scaffolding and must be masked out of the
        loss.
        """
        if audio_embeds.dim() != 3:
            raise ValueError(f"expected [B, A, D], got {tuple(audio_embeds.shape)}")
        b = audio_embeds.shape[0]
        pre = self._embed(self.prefix_ids, b)
        suf = self._embed(self.suffix_ids, b)
        wrapped = torch.cat([pre, audio_embeds.to(self.dtype), suf], dim=1)
        return wrapped, wrapped.shape[1]


def demo():
    import torch.nn as nn

    class Emb(nn.Module):
        def __init__(self, v, d):
            super().__init__()
            self.e = nn.Embedding(v, d)

        def forward(self, x):
            return self.e(x)

    class FakeBackbone(nn.Module):
        def __init__(self, d=16, v=64):
            super().__init__()
            self.config = type("C", (), {"hidden_size": d})()
            self.emb = Emb(v, d)

        def get_input_embeddings(self):
            return self.emb

    class FakeTok:
        chat_template = "yes"

        def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
            out = ""
            for m in msgs:
                out += f"<|{m['role']}|>{m['content']}<|end|>"
            return out + "<|assistant|>"

        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            ids = torch.tensor([[(ord(c) % 60) + 1 for c in text[:40]] or [1]])
            return type("R", (), {"input_ids": ids})()

    bb = FakeBackbone()
    w = ChatWrapper(FakeTok(), bb, system="be helpful")

    # The template must be split around the audio, not concatenated after it.
    assert w.prefix_ids.numel() > 0, "no prefix — audio would start the sequence"
    assert w.suffix_ids.numel() > 0, "no suffix — model never told it is its turn"
    assert w.overhead_tokens == w.prefix_ids.numel() + w.suffix_ids.numel()

    audio = torch.randn(2, 10, 16)
    wrapped, n_before_target = w.wrap(audio)
    assert wrapped.shape == (2, w.prefix_ids.numel() + 10 + w.suffix_ids.numel(), 16)
    assert n_before_target == wrapped.shape[1]

    # The audio must survive unchanged in the middle — the wrapper brackets it,
    # it does not transform it.
    p = w.prefix_ids.numel()
    assert torch.allclose(wrapped[:, p:p + 10], audio.to(wrapped.dtype), atol=1e-5)

    # Batch size follows the audio, and the template is shared across it.
    w5, _ = w.wrap(torch.randn(5, 3, 16))
    assert w5.shape[0] == 5
    assert torch.allclose(w5[0, :p], w5[4, :p]), "template must be identical per item"

    # A model with no chat template still gets a minimal one — bare embeddings
    # are what caused the fallback-to-instruct failure.
    class Bare(FakeTok):
        chat_template = None

    bare = ChatWrapper(Bare(), bb, system=None)
    assert bare.prefix_ids.numel() > 0 and bare.suffix_ids.numel() > 0

    # A template that swallows the placeholder must fail loudly rather than
    # silently putting audio in the wrong place.
    class Swallows(FakeTok):
        def apply_chat_template(self, msgs, **kw):
            return "no placeholder here"

    try:
        ChatWrapper(Swallows(), bb)
    except ValueError as exc:
        assert "placeholder" in str(exc)
    else:
        raise AssertionError("must reject a template that drops the placeholder")

    try:
        w.wrap(torch.randn(10, 16))
    except ValueError:
        pass
    else:
        raise AssertionError("un-batched audio must be rejected")

    print(f"prompt: template overhead {w.overhead_tokens} tokens, "
          f"audio spliced at position {p}")
    print("prompt: split, splice, batch sharing, guards OK")


if __name__ == "__main__":
    demo()
