"""Can the projector memorise 8 clips? The decisive capacity test.

If a projector cannot overfit eight utterances it will never learn eight
thousand, and no amount of corpus fixes it. Failing here means the bottleneck
is structural — capacity, objective, or encoder — not data.

Observed on the real runs: the projector emits a near-constant embedding
(cross-clip cosine 0.990) and the backbone produces the same generic reply for
every input. At scale, emitting a constant genuinely minimises loss better than
a weak audio signal does. This probe asks whether it CAN do better when
memorising is the easiest available strategy.

    python research/overfit_probe.py --corpus corpus/ --backbone <id>

Success is not a loss number. It is DIVERSITY: does the projector's output
differ across clips? A run that drives loss down while staying collapsed has
memorised the token distribution, not the audio.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from jaeger_agent_omni.capture import SAMPLE_RATE, CaptureReader  # noqa: E402
from jaeger_agent_omni.neural import JaegerAudioProjector          # noqa: E402
from jaeger_agent_omni.prompt import ChatWrapper                   # noqa: E402


def cross_clip_cosine(vecs: torch.Tensor) -> float:
    """Mean pairwise cosine between mean-pooled outputs. 1.0 = collapsed."""
    v = F.normalize(vecs, dim=-1)
    sim = v @ v.T
    off = sim[~torch.eye(len(v), dtype=bool, device=v.device)]
    return float(off.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--whisper", default="openai/whisper-tiny")
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-mult", type=int, default=1,
                    help="widen the projector: capacity lever")
    ap.add_argument("--no-embed-norm", action="store_true",
                    help="drop the output scale pin: does it cause collapse?")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from transformers import (AutoModelForCausalLM, AutoTokenizer, WhisperModel,
                              WhisperProcessor)

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")

    proc = WhisperProcessor.from_pretrained(args.whisper)
    whisper = WhisperModel.from_pretrained(args.whisper)
    enc = whisper.encoder.to(device).eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(args.backbone)
    lm = AutoModelForCausalLM.from_pretrained(args.backbone).to(device).eval()
    lm.requires_grad_(False)

    emb_norm = None if args.no_embed_norm else \
        lm.get_input_embeddings().weight.norm(dim=-1).mean().item()
    proj = JaegerAudioProjector(audio_dim=whisper.config.d_model,
                                cognition_dim=lm.config.hidden_size,
                                embed_norm=emb_norm).to(device)
    if args.hidden_mult > 1:
        # Widen the middle: 1024 -> H*cog -> cog instead of 1024 -> cog -> cog.
        import torch.nn as nn
        h = lm.config.hidden_size * args.hidden_mult
        proj.linear1 = nn.Linear(whisper.config.d_model, h).to(device)
        proj.linear2 = nn.Linear(h, lm.config.hidden_size).to(device)

    wrapper = ChatWrapper(tok, lm, device=device)
    reader = CaptureReader(args.corpus)
    turns = list(reader)[: args.clips]

    print(f"probe: {len(turns)} clips, {proj.n_params()/1e3:.0f}K projector params, "
          f"embed_norm={'off' if args.no_embed_norm else f'{emb_norm:.2f}'}, "
          f"hidden_mult={args.hidden_mult}")

    feats, labels = [], []
    for t in turns:
        f = proc(reader.audio(t), sampling_rate=SAMPLE_RATE,
                 return_tensors="pt").input_features[0]
        feats.append(f)
        ids = tok(t.transcript, return_tensors="pt").input_ids[0]
        labels.append(torch.cat([ids, torch.tensor([tok.eos_token_id])]))

    feats = torch.stack(feats).to(device)
    n = max(len(x) for x in labels)
    label_pad = torch.full((len(labels), n), -100, dtype=torch.long)
    for i, x in enumerate(labels):
        label_pad[i, : len(x)] = x
    label_pad = label_pad.to(device)
    durations = [t.duration_s for t in turns]

    with torch.no_grad():
        states = enc(feats).last_hidden_state.float()

    opt = torch.optim.AdamW(proj.parameters(), lr=args.lr)
    embed_layer = lm.get_input_embeddings()

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        audio = proj(states)
        wrapped, n_ctx = wrapper.wrap(audio)
        tgt = embed_layer(label_pad.clamp(min=0)).to(wrapped.dtype)
        inputs = torch.cat([wrapped, tgt], dim=1)
        mask = torch.full((len(turns), n_ctx), -100, dtype=torch.long, device=device)
        full = torch.cat([mask, label_pad], dim=1)

        logits = lm(inputs_embeds=inputs).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               full[:, 1:].reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
        opt.step()

        if step % max(1, args.steps // 8) == 0 or step == 1:
            with torch.no_grad():
                pooled = proj(states).mean(dim=1)
                div = cross_clip_cosine(pooled)
                pred = logits[:, n_ctx - 1:-1].argmax(-1)
                gold = label_pad
                m = gold != -100
                acc = ((pred[:, :gold.shape[1]] == gold) & m).sum().item() / m.sum().item()
            print(f"  step {step:>4}  loss {loss.item():.4f}  "
                  f"train_acc {acc:.3f}  cross-clip cosine {div:.4f}")

    print()
    if div > 0.98:
        print("COLLAPSED — the projector emits a near-constant regardless of input.")
        print("  Loss fell by memorising the token distribution, not the audio.")
        print("  Structural: capacity, objective or encoder. More data will not fix it.")
    elif acc > 0.9:
        print("MEMORISED — the projector can differentiate and reproduce these clips.")
        print("  The architecture works; the earlier plateau is scale, not structure.")
    else:
        print("PARTIAL — differentiating but not reproducing. Capacity-limited.")
    return 0


def demo():
    v = torch.randn(6, 32)
    assert 0.0 <= abs(cross_clip_cosine(v)) <= 1.0
    same = torch.ones(6, 32)
    assert cross_clip_cosine(same) > 0.999, "identical vectors must read as collapsed"
    orth = torch.eye(6, 32)
    assert abs(cross_clip_cosine(orth)) < 0.01, "orthogonal vectors must read as diverse"
    print("overfit_probe: collapse metric OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        sys.exit(main())
