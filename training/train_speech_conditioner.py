"""Distil MiniCPM Talker conditions into the tensor-only Jaeger bridge.

Two explicit phases keep evidence reproducible:

  capture  question PCM -> Gemma cache; teacher answer IDs are forced through
           Gemma only OFFLINE; paired Gemma tensors and MiniCPM's exact merged
           Talker conditions are saved.
  train    fit GemmaTalkerConditioner on train turns and report held-out turns.

Decoded text never appears in the saved pair file and the production module
has no string input.  The small 12-case corpus is a plumbing pilot, not a
general speech model; a production claim requires a much larger held-out set.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from jaeger_agent_omni.neural.speech import (  # noqa: E402
    GemmaTalkerConditioner, SpeechConditionerConfig)


def _load_matrix(path: Path) -> torch.Tensor:
    with path.open("rb") as f:
        rows, width = struct.unpack("<ii", f.read(8))
        raw = np.frombuffer(f.read(), dtype="<f4").copy()
    if raw.size != rows * width:
        raise ValueError(f"{path}: corrupt matrix payload")
    return torch.from_numpy(raw.reshape(rows, width))


def _teacher_turn(round_dir: Path) -> tuple[str, torch.Tensor]:
    chunks = sorted((round_dir / "llm_debug").glob("chunk_*"),
                    key=lambda p: int(p.name.split("_")[-1]))
    if not chunks:
        raise ValueError(f"{round_dir}: no teacher chunks")
    # Text exists only transiently here to ask Gemma for matching token-level
    # activations. It is not serialized in the pair corpus.
    text = "".join((p / "llm_text.txt").read_text() for p in chunks)
    merged = torch.cat([_load_matrix(p / "merged_embeddings.bin")
                        for p in chunks], dim=0)
    return text, merged


def _resample(sequence: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Length-align teacher tokenization to Gemma tokenization.

    Alignment is offline and monotonic. Linear interpolation avoids pretending
    the two vocabularies have one-to-one token boundaries while preserving the
    complete condition trajectory.
    """
    if sequence.shape[0] == n_tokens:
        return sequence
    x = sequence.T.unsqueeze(0)
    return F.interpolate(x, size=n_tokens, mode="linear",
                         align_corners=True).squeeze(0).T.contiguous()


def capture(args) -> int:
    import mlx.core as mx

    from jaeger_agent_omni.app import read_wav
    from jaeger_agent_omni.omni import build

    if args.manifest:
        cases = [json.loads(line) for line in args.manifest.read_text().splitlines()
                 if line.strip() and int(json.loads(line).get(
                     "condition_chunks", 1)) > 0]
    else:
        cases = json.loads((args.cases / "cases.json").read_text())
    session, _ = build(args.model)
    session.warmup()
    items = []
    for i, case in enumerate(cases):
        round_idx = int(case.get("round", i))
        round_dir = args.teacher / f"round_{round_idx:03d}"
        text, target = _teacher_turn(round_dir)
        ids = session.processor.tokenizer(
            text, add_special_tokens=False).input_ids

        session.reset()
        session.open_turn()
        audio_path = (Path(case["audio"]) if case.get("audio")
                      else args.cases / f"{case['key']}.wav")
        audio = read_wav(audio_path)
        for j in range(0, len(audio), 1600):
            session.feed_audio(audio[j:j + 1600])
        session.end_of_turn()
        token_emb, hidden = session.force_tokens(ids)
        # MLX exports bfloat16 with a PEP-3118 format NumPy does not
        # understand ("oat16"). Cast while it is still an MLX tensor.
        token = torch.from_numpy(np.array(
            token_emb.astype(mx.float32), copy=False)).float()[0]
        state = torch.from_numpy(np.stack([
            np.array(h.astype(mx.float32), copy=False).reshape(-1)
            for h in hidden])).float()
        target = _resample(target.float(), len(ids))
        items.append({"case": case["key"], "token_embeddings": token,
                      "hidden_states": state, "target_conditions": target})
        print(f"{case['key']} Gemma {len(ids):2d} tokens <- teacher "
              f"{sum(_load_matrix(p).shape[0] for p in round_dir.glob('llm_debug/chunk_*/merged_embeddings.bin')):2d}",
              flush=True)

    payload = {"format": "jaeger-condition-pairs-v1", "items": items,
               "model": args.model,
               "teacher": str(args.teacher.resolve())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"wrote {args.out}: {len(items)} turns, "
          f"{sum(len(x['hidden_states']) for x in items)} tokens")
    return 0


def _split(items: list[dict]) -> tuple[list[dict], list[dict]]:
    # Stable turn-level split: no tokens from a held-out utterance leak into
    # training. Every fifth item yields 9 train / 3 val on cases12.
    val = [x for i, x in enumerate(items) if i % 5 == 0]
    train = [x for i, x in enumerate(items) if i % 5 != 0]
    return train, val


def _stack(items: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(torch.cat([x[k] for x in items], dim=0).unsqueeze(0)
                 for k in ("token_embeddings", "hidden_states",
                           "target_conditions"))


@torch.no_grad()
def _metrics(model, batch) -> dict:
    token, hidden, target = batch
    pred = model(token, hidden)
    return {"loss": float(model.loss(pred, target)),
            "mse": float(F.mse_loss(pred, target)),
            "cosine": float(F.cosine_similarity(pred, target, dim=-1).mean())}


def train(args) -> int:
    pairs = torch.load(args.pairs, map_location="cpu", weights_only=False)
    if pairs.get("format") != "jaeger-condition-pairs-v1":
        raise ValueError("unknown pair corpus format")
    train_items, val_items = _split(pairs["items"])
    train_batch, val_batch = _stack(train_items), _stack(val_items)
    cdim = train_batch[1].shape[-1]
    tdim = train_batch[0].shape[-1]
    odim = train_batch[2].shape[-1]
    torch.manual_seed(args.seed)
    model = GemmaTalkerConditioner(SpeechConditionerConfig(
        cognition_dim=cdim, token_dim=tdim, talker_dim=odim,
        hidden_dim=args.hidden))
    if args.lexical_init:
        lexical = torch.load(args.lexical_init, map_location="cpu",
                             weights_only=False)
        if lexical.get("format") != "jaeger-lexical-bridge-v1":
            raise ValueError("unknown lexical bridge format")
        model.token_path.load_state_dict(lexical["state"])
        print(f"loaded lexical initialization: holdout cosine "
              f"{lexical.get('holdout_cosine', float('nan')):.3f}")
    if args.freeze_lexical:
        model.token_path.requires_grad_(False)
    # Start at the teacher condition's real scale, not unit norm.
    with torch.no_grad():
        initial = model(train_batch[0], train_batch[1])
        model.out_scale.mul_(train_batch[2].norm(dim=-1).mean() /
                             initial.norm(dim=-1).mean().clamp(min=1e-6))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    before = {"train": _metrics(model, train_batch),
              "val": _metrics(model, val_batch)}
    for step in range(1, args.steps + 1):
        model.train()
        pred = model(train_batch[0], train_batch[1])
        loss = model.loss(pred, train_batch[2])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.report_every == 0 or step == args.steps:
            vm = _metrics(model.eval(), val_batch)
            print(f"step {step:5d} train {float(loss):.5f}  "
                  f"val {vm['loss']:.5f} cos {vm['cosine']:.3f}", flush=True)
    after = {"train": _metrics(model.eval(), train_batch),
             "val": _metrics(model, val_batch)}
    ck = model.checkpoint()
    ck.update({"metrics_before": before, "metrics_after": after,
               "train_cases": [x["case"] for x in train_items],
               "val_cases": [x["case"] for x in val_items],
               "pairs": str(args.pairs.resolve())})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, args.out)
    print(json.dumps({"before": before, "after": after}, indent=2))
    print(f"wrote {args.out}")
    return 0


def selfcheck() -> int:
    x = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    assert _resample(x, 7).shape == (7, 3)
    items = [{"case": f"q{i}", "token_embeddings": torch.randn(2, 4),
              "hidden_states": torch.randn(2, 4),
              "target_conditions": torch.randn(2, 3)} for i in range(12)]
    tr, va = _split(items)
    assert len(tr) == 9 and len(va) == 3
    assert not ({x["case"] for x in tr} & {x["case"] for x in va})
    assert _stack(tr)[0].shape == (1, 18, 4)
    print("train_speech_conditioner: matrix IO, resample, held-out split OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--teacher", type=Path, required=True)
    cap.add_argument("--cases", type=Path, default=REPO / "research/cases12")
    cap.add_argument("--manifest", type=Path, default=None,
                     help="JSONL from research/capture_minicpm_teacher.py")
    cap.add_argument("--model", default=str(
        Path.home() / "models/gemma-4-e4b-it-mlx-full"))
    cap.add_argument("--out", type=Path, default=REPO / "training/runs/condition_pairs.pt")
    fit = sub.add_parser("train")
    fit.add_argument("--pairs", type=Path, required=True)
    fit.add_argument("--out", type=Path,
                     default=REPO / "training/runs/gemma_talker_conditioner.pt")
    fit.add_argument("--steps", type=int, default=1000)
    fit.add_argument("--hidden", type=int, default=512)
    fit.add_argument("--lr", type=float, default=2e-4)
    fit.add_argument("--weight-decay", type=float, default=0.01)
    fit.add_argument("--report-every", type=int, default=100)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--lexical-init", type=Path, default=None)
    fit.add_argument("--freeze-lexical", action="store_true",
                     help="preserve the broad word-level map on small corpora")
    sub.add_parser("selfcheck")
    args = ap.parse_args()
    if args.command == "capture":
        return capture(args)
    if args.command == "train":
        return train(args)
    return selfcheck()


if __name__ == "__main__":
    raise SystemExit(main())
