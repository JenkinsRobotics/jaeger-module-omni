"""Stage 5 — turn-end head. Our stand-in for MiniCPM's <|speak|>/<|listen|>.

MiniCPM decides turn-taking in token space: at chunk boundaries the backbone
predicts <|speak|> or <|listen|>, learned from full-duplex conversational data
(omni.h:460-469). We cannot copy that — it needs new vocabulary trained into
the backbone, which is not projector-scale work and Gemma has neither the tokens
nor the training.

So we copy the behaviour instead: a small binary head over the cognition model's
hidden state, asked once per 1 s chunk —

    "is the user still speaking, or is it my turn?"

Labels come free from `capture.py`: every logged turn boundary is a positive
example, every chunk before it a negative one. That data is accumulating already.

Why not just a silence timeout: a timeout cannot tell "...and then he said" from
"...and that was that." Both are ~400 ms of silence; only one is a handover. The
head sees the hidden state, so it can use the semantics. That difference is what
separates conversational from merely fast.

    python -m jaeger_agent_omni.core.neural.heads     # self-check
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Asymmetric by design. Interrupting a speaker mid-sentence is a worse failure
# than waiting one extra chunk (100 ms of dead air vs stepping on their words),
# so the default threshold is deliberately above 0.5.
DEFAULT_THRESHOLD = 0.7


class TurnEndHead(nn.Module):
    """Binary classifier over pooled cognition hidden state.

    Deliberately tiny — it rides on a frozen backbone that already did the
    understanding. If this needs to be large, the backbone is not being used.
    """

    def __init__(self, cognition_dim: int = 2048, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cognition_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[B, T, D] -> [B] logits. Mean-pools over the chunk's tokens."""
        if hidden_states.dim() != 3:
            raise ValueError(f"expected [B, T, D], got {tuple(hidden_states.shape)}")
        return self.net(hidden_states.mean(dim=1)).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TurnEndDetector:
    """Stateful wrapper: hysteresis over consecutive chunks.

    A single confident chunk is not enough — mid-sentence pauses spike the
    score. Requiring `patience` consecutive chunks above threshold costs at most
    (patience-1) x 100 ms and removes most false handovers.
    """

    def __init__(self, head: TurnEndHead, threshold: float = DEFAULT_THRESHOLD,
                 patience: int = 2):
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold {threshold} must be in (0, 1)")
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.head = head
        self.threshold = threshold
        self.patience = patience
        self._streak = 0
        self.last_prob = 0.0

    @torch.no_grad()
    def step(self, hidden_states: torch.Tensor) -> bool:
        """One chunk in, "is it my turn" out."""
        prob = torch.sigmoid(self.head(hidden_states)).item()
        self.last_prob = prob
        self._streak = self._streak + 1 if prob >= self.threshold else 0
        return self._streak >= self.patience

    def reset(self):
        """Call at the start of every user turn."""
        self._streak = 0
        self.last_prob = 0.0


def turn_end_loss(logits: torch.Tensor, labels: torch.Tensor,
                  false_handover_weight: float = 3.0) -> torch.Tensor:
    """Weighted BCE. Interrupting the user costs more than waiting.

    labels: 1.0 = turn ended, 0.0 = user still speaking.
    A false positive (predicting 1 when the user is mid-sentence) is the
    expensive mistake, so it is upweighted.
    """
    weight = torch.where(labels < 0.5, torch.full_like(labels, false_handover_weight),
                         torch.ones_like(labels))
    return F.binary_cross_entropy_with_logits(logits, labels, weight=weight)


def demo():
    torch.manual_seed(0)
    cognition_dim = 2048
    head = TurnEndHead(cognition_dim=cognition_dim)

    # Tiny: it rides on a frozen backbone that already did the work.
    assert head.n_params() < 1_000_000, head.n_params()

    # One chunk of audio tokens (1 s -> 10 tokens) -> one decision per batch item.
    assert head(torch.randn(4, 10, cognition_dim)).shape == (4,)
    # Length-agnostic, so it works on partial chunks too.
    assert head(torch.randn(1, 3, cognition_dim)).shape == (1,)

    try:
        head(torch.randn(10, cognition_dim))
    except ValueError:
        pass
    else:
        raise AssertionError("must reject un-batched input")

    # Hysteresis: one confident chunk must not trigger a handover.
    det = TurnEndDetector(head, threshold=0.5, patience=2)
    with torch.no_grad():
        head.net[-1].bias.fill_(10.0)            # force prob ~ 1
    x = torch.randn(1, 10, cognition_dim)
    assert det.step(x) is False, "single chunk must not hand over"
    assert det.step(x) is True, "two consecutive should"
    assert det.last_prob > 0.99

    # A mid-sentence dip breaks the streak — this is the pause case.
    det.reset()
    assert det.step(x) is False
    with torch.no_grad():
        head.net[-1].bias.fill_(-10.0)           # user resumes talking
    assert det.step(x) is False
    with torch.no_grad():
        head.net[-1].bias.fill_(10.0)
    assert det.step(x) is False, "streak must restart after a dip"
    assert det.step(x) is True

    # reset() is what a new user turn calls.
    det.reset()
    assert det._streak == 0 and det.last_prob == 0.0

    for bad in (dict(threshold=0.0), dict(threshold=1.0), dict(patience=0)):
        try:
            TurnEndDetector(head, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should have rejected {bad}")

    # Loss asymmetry: the same magnitude of error costs more when the model
    # wrongly claims the turn than when it wrongly waits.
    logit = torch.tensor([2.0])
    interrupt = turn_end_loss(logit, torch.tensor([0.0]))   # said "my turn", wasn't
    wait = turn_end_loss(-logit, torch.tensor([1.0]))       # said "wait", shouldn't
    assert interrupt > wait, (interrupt.item(), wait.item())
    assert torch.isclose(interrupt / wait, torch.tensor(3.0), atol=1e-4)

    # Trainable.
    loss = turn_end_loss(head(torch.randn(2, 10, cognition_dim)), torch.tensor([1.0, 0.0]))
    loss.backward()
    assert head.net[0].weight.grad.abs().sum() > 0

    print(f"turn_end: {head.n_params()/1e3:.0f}K params, "
          f"threshold {DEFAULT_THRESHOLD}, patience 2 = {2 * 100}ms worst-case delay")
    print("turn_end: shapes, hysteresis, reset, asymmetric loss, grads OK")


if __name__ == "__main__":
    demo()
