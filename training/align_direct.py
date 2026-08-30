"""Train the adapter WITHOUT the backbone. Minutes, not hours.

The adapter's job is format conversion: put whisper's output where Gemma's
token embeddings live. That is checkable against the embedding TABLE alone —
no 14 GB backbone, no generation, no 42 layers of activations.

    adapter output [T, 2560]  @  table.T [2560, V]  ->  logits [T, V]
    CTC against the transcript tokens

CTC because the sequences do not align: 5 s of audio is 50 adapter vectors but
maybe 12 text tokens, and nobody knows which vector belongs to which word. CTC
sums over every valid alignment instead of requiring one.

This is exactly `research/adapter_readout.py` turned into a loss. The default
warm-start preserves the requested architecture and values:

    MiniCPM-tuned Whisper -> ONE MiniCPM-shaped adapter -> Gemma

MiniCPM linear1 is copied and its linear2 is composed with the Qwen->Gemma
ridge map, then the whole one-adapter module is trainable. There is no frozen
Qwen-shaped output followed by a second delta. A random/stock-Whisper start is
available only as an explicit ablation.

The older trained-through-the-backbone probe emitted

    'The capital of France is Paris.' -> introductions niOkno ิ ิ activity ...

with cosine -0.20 to the transcript: right space, right scale, wrong
direction, and barely varying with the input. An hour of backbone training
produced that; this objective attacks it head-on.

    python training/align_direct.py --corpus training/corpus_mixed --steps 2000
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "research"))

GEMMA = "google/gemma-4-E4B-it"
BLANK = 0
DEFAULT_ALIGNMENT = (REPO.parent / "VoiceLLM-Playground" / "Omni Model" /
                     "gemma-backbone" / "W_fwd.npy")


def annotate(row: dict) -> str:
    """Transcript plus INLINE annotations, the way whisper already reports
    non-speech events in its own stream:

        " Hello there [LAUGHTER] as I was saying"
        " the capital of france is paris [end]"

    No separate output heads and no special slots. These are ordinary words,
    so a FROZEN Gemma reads them with the understanding it already has —
    which a novel token at a novel position would not get.

    turn:    [end] when the utterance completed, nothing when it was cut off
             mid-word. That is the free supervision truncation already gives.
    speaker: [speaker N] when the corpus knows who was talking.
    """
    text = (row.get("transcript") or "").strip()
    meta = row.get("meta") or {}
    spk = str(meta.get("speaker") or row.get("speaker") or "").strip()
    parts = []
    if spk:
        parts.append(f"[speaker {spk}]")
    parts.append(text)
    # `turn_ended` defaults to 1: a corpus turn is a whole utterance unless
    # the loader truncated it.
    if float(row.get("turn_ended", 1.0)) >= 0.5:
        parts.append("[end]")
    return " ".join(parts)


def build_targets(tok, rows: list):
    import torch
    seqs = [torch.tensor(tok(annotate(r), add_special_tokens=False).input_ids)
            for r in rows]
    lens = torch.tensor([len(s) for s in seqs])
    return torch.cat(seqs), lens


def main() -> int:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperModel

    from fit_alignment import embedding_table
    from jaeger_agent_omni.core.neural.whisper_to_gemma import WhisperToGemma

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--whisper", default="openai/whisper-medium.en")
    ap.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    ap.add_argument("--init", choices=("minicpm_remap", "random"),
                    default="minicpm_remap",
                    help="default preserves MiniCPM's tuned Whisper/projector values")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="training/runs/adapter_direct.pt")
    a = ap.parse_args()

    dev = a.device
    tok = AutoTokenizer.from_pretrained(GEMMA)
    table = embedding_table(GEMMA)
    scale = table.shape[1] ** 0.5
    table = (table * scale).to(dev)                # what Gemma actually receives
    emb_norm = float(table.norm(dim=-1).mean())
    print(f"Gemma table {tuple(table.shape)}  emb_norm {emb_norm:.2f}")

    # The production default is the user-requested graph: MiniCPM's tuned
    # Whisper -> ONE MiniCPM-shaped adapter retargeted into Gemma. There is no
    # frozen 4096 output followed by an extra delta. `random` remains only as
    # an explicit ablation.
    if a.init == "minicpm_remap":
        import numpy as np
        from load_minicpm import load_front_end
        encoder, minicpm_projector = load_front_end()
        mapping = np.load(a.alignment)
        model = WhisperToGemma.from_minicpm(
            minicpm_projector, mapping, cognition_dim=table.shape[1],
            embed_norm=emb_norm)
        encoder = encoder.to(dev).eval().requires_grad_(False)
        audio_dim = 1024
        print("encoder: MiniCPM-tuned Whisper, frozen")
        print("adapter init: MiniCPM linear1 copied + linear2 remapped to Gemma")
    else:
        wm = WhisperModel.from_pretrained(a.whisper)
        encoder = wm.encoder.to(dev).eval().requires_grad_(False)
        audio_dim = wm.config.d_model
        del wm
        model = WhisperToGemma(audio_dim=audio_dim,
                               cognition_dim=table.shape[1],
                               embed_norm=emb_norm)
        print(f"encoder: stock Whisper {audio_dim}-dim, frozen (ablation)")
    model = model.to(dev).train()
    print(f"adapter {model.n_params()/1e6:.1f}M params, all trainable")

    rows = [json.loads(l) for l in
            (Path(a.corpus) / "turns.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if (r.get("transcript") or "").strip()]
    print(f"{len(rows)} turns")

    fx = WhisperFeatureExtractor.from_pretrained(a.whisper)
    from jaeger_agent_omni.app import read_wav

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    ctc = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)
    g = torch.Generator().manual_seed(0)
    t0, hist = time.monotonic(), []

    for step in range(1, a.steps + 1):
        idx = torch.randint(0, len(rows), (a.batch,), generator=g).tolist()
        batch = [rows[i] for i in idx]
        mels, durs = [], []
        for r in batch:
            audio = read_wav(Path(a.corpus) / r["audio_path"])
            mels.append(fx(audio, sampling_rate=16000,
                           return_tensors="pt").input_features[0])
            durs.append(len(audio) / 16000)
        mel = torch.stack(mels).to(dev)

        with torch.no_grad():
            h = encoder(mel).last_hidden_state
        out = model(h.float(), duration_s=max(durs))        # [B, T, 2560]

        logits = out @ table.T                              # [B, T, V]
        logp = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)  # [T,B,V]
        targets, tlens = build_targets(tok, batch)
        ilens = torch.full((len(batch),), out.shape[1], dtype=torch.long)
        loss = ctc(logp, targets.to(dev), ilens, tlens)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % max(1, a.steps // 20) == 0 or step == 1:
            el = time.monotonic() - t0
            hist.append({"step": step, "loss": float(loss)})
            print(f"  step {step:>5}  ctc {float(loss):8.4f}  "
                  f"{el/step:.2f}s/step", flush=True)

        # Save periodically, not only at the end. Without this a run cannot be
        # inspected until it finishes -- which means waiting out a run that
        # plateaued at step 140, with no way to read out what it learned.
        if step % max(1, a.steps // 4) == 0:
            torch.save({"model": model.state_dict(),
                        "cognition_dim": table.shape[1], "embed_norm": emb_norm,
                        "arch": "whisper_to_gemma", "backbone_id": GEMMA,
                        "history": hist, "step": step}, a.out)

    torch.save({"model": model.state_dict(), "cognition_dim": table.shape[1],
                "embed_norm": emb_norm, "arch": "whisper_to_gemma",
                "backbone_id": GEMMA, "history": hist, "step": a.steps},
               a.out)
    print(f"\nwrote {a.out}")
    print("check it:  python research/adapter_readout.py --ckpt " + a.out)
    return 0


def demo() -> None:
    """CTC must fall when the emission matches the target order."""
    import torch
    import torch.nn.functional as F

    T, V = 12, 30
    target = torch.tensor([3, 7, 5])
    ctc = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)
    torch.manual_seed(0)
    bad = F.log_softmax(torch.randn(T, 1, V), dim=-1)
    good = torch.full((T, 1, V), -12.0)
    good[:, 0, BLANK] = 0.0
    for t, tokid in zip((2, 6, 9), target.tolist()):
        good[t, 0, BLANK] = -12.0
        good[t, 0, tokid] = 0.0
    good = F.log_softmax(good, dim=-1)
    lb = ctc(bad, target, torch.tensor([T]), torch.tensor([3]))
    lg = ctc(good, target, torch.tensor([T]), torch.tensor([3]))
    assert lg < lb, (float(lg), float(lb))
    print(f"align_direct: CTC discriminates ({float(lg):.3f} aligned "
          f"< {float(lb):.3f} random)")


if __name__ == "__main__":
    raise SystemExit(demo() if "--demo" in sys.argv else main())
