"""Stage 3/4 — self-distillation losses.

The legacy stack is the teacher. Text-path and audio-path run the SAME frozen
backbone; only the input differs. So we are not teaching the backbone anything —
we are teaching the projector to land where the text already lands.

    text  : transcript -> tokenizer -> embeddings -> backbone -> H_text, logits_text
    audio : waveform -> whisper -> projector -> backbone -> H_audio, logits_audio

    loss = w_h * hidden(H_audio, H_text)     dense: every layer, every position
         + w_k * KL(audio || text)           output distribution agreement
         + w_c * CE(audio, response)         still answer correctly

Stage 3 (warmup) uses CE alone — audio->transcript, cheap and stable, lands the
projector in roughly the right region. Stage 4 turns on hidden + KL, which is
where alignment actually happens. Presets below.

Why hidden-state matching and not just next-token: the dense signal is the whole
reason this is cheap. One sample yields (layers x positions x dims) constraints
instead of one token of supervision.

    python training/distill.py     # self-check, no model download
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DistillWeights:
    hidden: float = 0.0
    kl: float = 0.0
    ce: float = 1.0
    temperature: float = 2.0

    @staticmethod
    def warmup() -> "DistillWeights":
        """Stage 3: transcript prediction only. Cheap, stable, gets in the door."""
        return DistillWeights(hidden=0.0, kl=0.0, ce=1.0)

    @staticmethod
    def distill() -> "DistillWeights":
        """Stage 4: internal-state alignment. Where understanding comes from."""
        return DistillWeights(hidden=1.0, kl=0.5, ce=0.5)


def hidden_state_loss(student: tuple, teacher: tuple,
                      layers: tuple = None) -> torch.Tensor:
    """Cosine distance between hidden states at selected layers.

    Cosine, not MSE: we want the audio representation pointing the same
    direction as the text one, and magnitude is not the part the backbone reads.
    MSE also lets deep layers with large activations dominate the gradient.

    Sequence lengths differ (audio tokens vs text tokens), so this compares
    sequence-mean vectors. Position-wise alignment would need the two streams to
    correspond token for token, which they do not.
    """
    layers = layers or tuple(range(len(teacher)))
    if len(student) != len(teacher):
        raise ValueError(f"layer count mismatch: {len(student)} vs {len(teacher)}")

    total = 0.0
    for i in layers:
        s = student[i].mean(dim=1)                  # [B, D]
        t = teacher[i].mean(dim=1).detach()         # teacher never gets gradient
        total = total + (1.0 - F.cosine_similarity(s, t, dim=-1)).mean()
    return total / len(layers)


def kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
            temperature: float = 2.0) -> torch.Tensor:
    """KL(teacher || student) over the last position, temperature-softened.

    Compares only the final position because that is where the two streams are
    aligned: both have just consumed the whole question and are about to answer.
    """
    s = F.log_softmax(student_logits[:, -1] / temperature, dim=-1)
    t = F.softmax(teacher_logits[:, -1].detach() / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)


def response_loss(logits: torch.Tensor, labels: torch.Tensor,
                  ignore_index: int = -100) -> torch.Tensor:
    """Standard shifted causal LM cross-entropy on the response tokens."""
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=ignore_index)


def total_loss(w: DistillWeights, *, student_hidden=None, teacher_hidden=None,
               student_logits=None, teacher_logits=None,
               labels=None) -> tuple:
    """Combine the active terms. Returns (loss, parts) — parts for logging.

    Terms with zero weight are skipped rather than multiplied by zero, so a
    warmup run never pays for a teacher forward pass it does not use.
    """
    parts = {}
    loss = torch.zeros((), device=_device_of(student_logits, student_hidden))

    if w.hidden > 0:
        if student_hidden is None or teacher_hidden is None:
            raise ValueError("hidden weight > 0 but hidden states not provided")
        parts["hidden"] = hidden_state_loss(student_hidden, teacher_hidden)
        loss = loss + w.hidden * parts["hidden"]

    if w.kl > 0:
        if student_logits is None or teacher_logits is None:
            raise ValueError("kl weight > 0 but logits not provided")
        parts["kl"] = kl_loss(student_logits, teacher_logits, w.temperature)
        loss = loss + w.kl * parts["kl"]

    if w.ce > 0:
        if student_logits is None or labels is None:
            raise ValueError("ce weight > 0 but logits/labels not provided")
        parts["ce"] = response_loss(student_logits, labels)
        loss = loss + w.ce * parts["ce"]

    if not parts:
        raise ValueError("all loss weights are zero")
    return loss, {k: v.item() for k, v in parts.items()}


def _device_of(*tensors):
    for t in tensors:
        if isinstance(t, torch.Tensor):
            return t.device
        if isinstance(t, (tuple, list)) and t:
            return t[0].device
    return "cpu"


def agreement_rate(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> float:
    """Fraction of items where both paths pick the same next token.

    This is the Stage 4 metric: does the audio path reach the same conclusion
    the text path did. Simple, and it is what the milestone check reads.
    """
    s = student_logits[:, -1].argmax(-1)
    t = teacher_logits[:, -1].argmax(-1)
    return (s == t).float().mean().item()


def demo():
    torch.manual_seed(0)
    b, v, d, n_layers = 4, 32, 16, 3

    teacher_h = tuple(torch.randn(b, 9, d) for _ in range(n_layers))
    student_h = tuple(torch.randn(b, 5, d) for _ in range(n_layers))  # shorter seq

    # Different sequence lengths must be fine — audio and text tokenise
    # differently and never line up 1:1.
    hl = hidden_state_loss(student_h, teacher_h)
    assert hl.shape == (), hl.shape
    assert 0.0 <= hl.item() <= 2.0

    # Identical representations => zero loss. Anchors the whole objective.
    assert hidden_state_loss(teacher_h, teacher_h).item() < 1e-6

    # Closer representations => strictly lower loss.
    near = tuple(t + 0.01 * torch.randn_like(t) for t in teacher_h)
    far = tuple(torch.randn_like(t) for t in teacher_h)
    assert hidden_state_loss(near, teacher_h) < hidden_state_loss(far, teacher_h)

    try:
        hidden_state_loss(student_h[:2], teacher_h)
    except ValueError:
        pass
    else:
        raise AssertionError("layer-count mismatch must be caught")

    # KL: zero against itself, positive otherwise.
    tl = torch.randn(b, 6, v)
    assert kl_loss(tl, tl).item() < 1e-6
    assert kl_loss(torch.randn(b, 6, v), tl).item() > 0

    # The teacher must never receive gradient — it is the same frozen weights.
    t_grad = tuple(t.clone().requires_grad_(True) for t in teacher_h)
    s_grad = tuple(t.clone().requires_grad_(True) for t in student_h)
    hidden_state_loss(s_grad, t_grad).backward()
    assert all(t.grad is None for t in t_grad), "teacher was not detached"
    assert all(s.grad is not None for s in s_grad)

    # Presets behave as documented.
    warm = DistillWeights.warmup()
    assert warm.hidden == 0.0 and warm.kl == 0.0 and warm.ce == 1.0
    full = DistillWeights.distill()
    assert full.hidden > 0 and full.kl > 0 and full.ce > 0

    logits = torch.randn(b, 6, v, requires_grad=True)
    labels = torch.randint(0, v, (b, 6))

    # Warmup needs no teacher at all — skipping unused terms is the point.
    loss, parts = total_loss(warm, student_logits=logits, labels=labels)
    assert set(parts) == {"ce"}, parts
    loss.backward()
    assert logits.grad is not None

    loss, parts = total_loss(full, student_hidden=student_h, teacher_hidden=teacher_h,
                             student_logits=logits, teacher_logits=tl, labels=labels)
    assert set(parts) == {"hidden", "kl", "ce"}, parts

    # Missing inputs for an active term must fail loudly, not silently skip.
    try:
        total_loss(full, student_logits=logits, labels=labels)
    except ValueError:
        pass
    else:
        raise AssertionError("must reject active term with missing inputs")

    try:
        total_loss(DistillWeights(0, 0, 0), student_logits=logits, labels=labels)
    except ValueError:
        pass
    else:
        raise AssertionError("all-zero weights must be rejected")

    # Agreement rate is the Stage 4 metric.
    assert agreement_rate(tl, tl) == 1.0
    assert 0.0 <= agreement_rate(torch.randn(b, 6, v), tl) <= 1.0

    print(f"distill: warmup={warm}, distill={full}")
    print("distill: cosine floor, monotonicity, detach, presets, guards OK")


if __name__ == "__main__":
    demo()
