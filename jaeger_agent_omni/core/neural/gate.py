"""Audio gates — trigger on a remembered sound. No training, no second model.

Hear a phrase once, keep its embedding, and trigger when it comes round again.
That is all a wake word is, and it needs no trained detector:

    enroll   reference audio -> encoder -> normalised embedding, stored
    listen   incoming chunk  -> encoder -> cosine similarity -> threshold

The system is **always** concurrent full duplex. A gate does not change the
mode — it decides whether what was heard should act. Everything upstream keeps
running either way.

The architectural point: this consumes the **same encoder output the projector
consumes**. One encode, two consumers. Adding a gate costs a dot product, not
a model.

    python -m jaeger_agent_omni.core.neural.gate     # self-check

## What this is honestly good at

Matching a specific phrase said in a similar way — a character catchphrase, a
recorded wake word, a sound effect. Enroll several takes and it tolerates more.

## What it is not

Speaker-independent. Whisper's encoder is trained for ASR so its states are
content-heavy, which helps, but a phrase enrolled in one voice will match that
voice best. For anyone-can-say-it triggers, enroll several speakers or gate on
the transcript instead.
"""

import torch
import torch.nn.functional as F


def embed(states: torch.Tensor) -> torch.Tensor:
    """Encoder states -> one normalised vector.

    Mean-pool then L2-normalise, so comparison is cosine and clip length does
    not change the scale.
    """
    if states.dim() == 2:
        states = states.unsqueeze(0)
    if states.dim() != 3:
        raise ValueError(f"expected [T, D] or [B, T, D], got {tuple(states.shape)}")
    return F.normalize(states.float().mean(dim=1), dim=-1)


class AudioGate:
    """Trigger on remembered sounds.

    `threshold` is the knob and it is deliberately exposed rather than tuned
    here: a wake word wants few false triggers (high), a sound-effect cue wants
    few misses (lower). Calibrate against real recordings, not intuition.
    """

    def __init__(self, threshold: float = 0.85):
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold {threshold} must be in (0, 1)")
        self.threshold = threshold
        self._refs: dict = {}

    def enroll(self, name: str, states: torch.Tensor) -> int:
        """Remember a sound under `name`. Call repeatedly to add takes.

        Several takes of the same phrase is the cheapest robustness available —
        each one covers a different speaking rate, distance or mood.
        """
        vec = embed(states)
        self._refs.setdefault(name, []).append(vec.squeeze(0).detach())
        return len(self._refs[name])

    def forget(self, name: str):
        self._refs.pop(name, None)

    @property
    def names(self) -> list:
        return sorted(self._refs)

    def scores(self, states: torch.Tensor) -> dict:
        """Best cosine similarity per enrolled name.

        Max over takes, not mean: matching *any* enrolled rendition should
        count. Averaging would let a poor take veto a good one.
        """
        if not self._refs:
            return {}
        query = embed(states).squeeze(0)
        return {name: max(float(query @ ref) for ref in refs)
                for name, refs in self._refs.items()}

    def match(self, states: torch.Tensor) -> tuple:
        """(name, score) of the best match above threshold, else (None, score)."""
        scores = self.scores(states)
        if not scores:
            return None, 0.0
        name = max(scores, key=scores.get)
        best = scores[name]
        return (name if best >= self.threshold else None), best


def demo():
    torch.manual_seed(0)
    d, t = 64, 50

    # Stand-ins for encoder output. A "phrase" is a consistent direction with
    # per-take variation, which is what repeated utterances actually look like.
    def utterance(base, noise=0.15):
        return base + noise * torch.randn(t, d)

    hey_jaeger = torch.randn(1, d).repeat(t, 1)
    other_words = torch.randn(1, d).repeat(t, 1)

    gate = AudioGate(threshold=0.85)
    assert gate.names == [] and gate.match(utterance(hey_jaeger)) == (None, 0.0)

    assert gate.enroll("hey_jaeger", utterance(hey_jaeger)) == 1
    assert gate.enroll("hey_jaeger", utterance(hey_jaeger)) == 2
    assert gate.names == ["hey_jaeger"]

    # A fresh take of the enrolled phrase triggers; unrelated audio does not.
    name, score = gate.match(utterance(hey_jaeger))
    assert name == "hey_jaeger", (name, score)
    assert score > 0.85, score

    name, score = gate.match(utterance(other_words))
    assert name is None, (name, score)

    # Several takes must beat one: more coverage of how the phrase gets said.
    # Threshold is irrelevant here — only the raw score is being compared.
    one, many = AudioGate(threshold=0.01), AudioGate(threshold=0.01)
    one.enroll("w", utterance(hey_jaeger, noise=0.05))
    for _ in range(6):
        many.enroll("w", utterance(hey_jaeger, noise=0.4))
    hard = utterance(hey_jaeger, noise=0.4)
    assert many.match(hard)[1] >= one.match(hard)[1] - 0.05

    # Normalisation: length must not change the score. A 1 s and a 3 s
    # rendition of the same phrase should score alike.
    g = AudioGate(threshold=0.01)
    g.enroll("w", hey_jaeger)
    short = g.match(hey_jaeger[:10])[1]
    long_ = g.match(hey_jaeger.repeat(3, 1))[1]
    assert abs(short - long_) < 1e-4, (short, long_)

    # Multiple gates coexist and the strongest wins.
    multi = AudioGate(threshold=0.5)
    multi.enroll("wake", utterance(hey_jaeger))
    multi.enroll("stop", utterance(other_words))
    assert multi.match(utterance(other_words))[0] == "stop"
    assert set(multi.scores(utterance(hey_jaeger))) == {"wake", "stop"}

    multi.forget("stop")
    assert multi.names == ["wake"]

    # Accepts a single clip's [T, D] as well as batched.
    assert embed(torch.randn(t, d)).shape == (1, d)
    assert torch.allclose(embed(torch.randn(t, d)).norm(), torch.tensor(1.0), atol=1e-5)

    for bad in (torch.randn(5), torch.randn(2, 3, 4, 5)):
        try:
            embed(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("bad shape must be rejected")

    for bad in (0.0, 1.0, -0.5):
        try:
            AudioGate(threshold=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"threshold {bad} must be rejected")

    print("neural.gate: enroll, match, multi-take, length invariance OK")


if __name__ == "__main__":
    demo()
