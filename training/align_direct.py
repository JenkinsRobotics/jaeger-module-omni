"""Train the adapter WITHOUT the backbone. Minutes, not hours.

The adapter's job is format conversion: put whisper's output where Gemma's
token embeddings live. That is checkable against the embedding TABLE alone —
no 14 GB backbone, no generation, no 42 layers of activations.

    adapter output [T, 2560]  @  table.T [2560, V]  ->  logits [T, V]
    CTC against the transcript tokens

CTC because the sequences do not align: 5 s of audio is 50 adapter vectors but
maybe 12 text tokens, and nobody knows which vector belongs to which word. CTC
sums over every valid alignment instead of requiring one.

This is exactly `research/adapter_readout.py` turned into a loss. That probe
showed the trained-through-the-backbone adapter emitting

    'The capital of France is Paris.' -> introductions niOkno ิ ิ activity ...

with cosine -0.20 to the transcript: right space, right scale, wrong
direction, and barely varying with the input. An hour of backbone training
produced that; this objective attacks it head-on.

    python training/align_direct.py --corpus training/corpus_mixed --steps 2000
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

GEMMA = "google/gemma-4-E4B-it"
BLANK = 0


def build_targets(tok, transcripts: list):
    import torch
    seqs = [torch.tensor(tok(t, add_special_tokens=False).input_ids)
            for t in transcripts]
    lens = torch.tensor([len(s) for s in seqs])
    return torch.cat(seqs), lens


def main() -> int:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperModel

    from fit_alignment import embedding_table
    from jaeger_agent_omni.neural.whisper_to_gemma import WhisperToGemma
    from load_minicpm import load_front_end

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--whisper", default="openai/whisper-medium.en")
    ap.add_argument("--alignment", default="training/runs/alignment.npz")
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

    encoder, front = load_front_end()
    encoder = encoder.to(dev).eval().requires_grad_(False)

    W = None
    if Path(a.alignment).exists():
        import numpy as np
        W = torch.tensor(np.load(a.alignment)["W_fwd"])
        print(f"seeding from ridge map {tuple(W.shape)}")
        model = WhisperToGemma.from_minicpm(front, W, cognition_dim=table.shape[1],
                                            embed_norm=emb_norm)
    else:
        print("NO ridge map — random linear2. Run training/fit_alignment.py "
              "first; cold start is what produced cosine -0.20.")
        model = WhisperToGemma(cognition_dim=table.shape[1], embed_norm=emb_norm)
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
        targets, tlens = build_targets(tok, [r["transcript"] for r in batch])
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
