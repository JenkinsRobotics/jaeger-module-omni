#!/usr/bin/env python3
"""Raw-audio causal overlap trial, independent of Whisper, adapter and Gemma.

prepare is CPU-only. train uses one GPU, selecting epochs and threshold on
validation only; test is evaluated once after selection. No runtime promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
ROOT = Path.home() / "models/jaeger-training-data/gemmaomni-cache-v1"
CACHE = Path.home() / "models/jaeger-training-data/parallel-overlap-v1"
HOP = 0.04


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fold(identity):
    return int(hashlib.sha256(str(identity).encode()).hexdigest()[:8], 16) % 6


def split_for(identities):
    folds = {fold(x) for x in identities}
    if not folds:
        raise ValueError("source identities are required")
    if folds == {0}:
        return "test"
    if folds == {1}:
        return "validation"
    return "train" if not folds & {0, 1} else None


def logmel(audio):
    """Causal 25 Hz features; each frame sees audio only through its right edge."""
    import torchaudio

    wave = torch.as_tensor(np.asarray(audio, np.float32))
    # Left padding makes frame k available after (k+1)*160 samples.
    spectrum = torch.stft(F.pad(wave, (352, 0)), n_fft=512,
                          hop_length=160, win_length=400,
                          window=torch.hann_window(400), center=False,
                          return_complex=True).abs().square()
    bank = torchaudio.functional.melscale_fbanks(257, 0, 8000, 80, 16000)
    mel = torch.log10(torch.clamp(bank.T @ spectrum, min=1e-10))
    usable = mel.shape[1] // 4 * 4
    return mel[:, :usable].reshape(80, -1, 4).mean(-1).T.numpy()


class OverlapNet(nn.Module):
    """Small causal TCN; per-frame normalization cannot leak future audio."""
    def __init__(self, width=128):
        super().__init__()
        self.input = nn.Linear(80, width)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(5)])
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 3, dilation=2**i)
                                   for i in range(5)])
        self.output = nn.Linear(width, 1)

    def forward(self, x):
        x = F.gelu(self.input((x + 3.0) / 3.0))
        for i, (norm, conv) in enumerate(zip(self.norms, self.convs)):
            h = norm(x).transpose(1, 2)
            x = x + F.gelu(conv(F.pad(h, (2 * 2**i, 0))).transpose(1, 2))
        return self.output(x).squeeze(-1)


def intervals(binary):
    padded = np.pad(np.asarray(binary, dtype=np.int8), (1, 1))
    return list(zip(np.where(np.diff(padded) == 1)[0],
                    np.where(np.diff(padded) == -1)[0]))


def metrics(predictions, threshold):
    tp = fp = fn = false_events = true_events = detected = frames = 0
    delays = []
    for prob, labels in predictions:
        truth, pred = labels >= 0.5, prob >= threshold
        tp += int((truth & pred).sum()); fp += int((~truth & pred).sum())
        fn += int((truth & ~pred).sum()); frames += len(labels)
        actual, guessed = intervals(truth), intervals(pred)
        true_events += len(actual)
        # One detection per truth event; repeated bursts count as false alarms.
        used = set()
        for start, end in guessed:
            matches = [i for i, (a, b) in enumerate(actual)
                       if i not in used and max(a, start) < min(b, end)]
            if not matches:
                false_events += 1
            else:
                i = matches[0]; used.add(i); detected += 1
                delays.append(max(0, start - actual[i][0]) * HOP * 1000)
    return {"f1": 2*tp/max(1, 2*tp+fp+fn), "tp": tp, "fp": fp, "fn": fn,
            "event_recall": detected/max(1, true_events),
            "events": true_events, "detected_events": detected,
            "false_alarm_events": false_events,
            "false_alarms_per_minute": false_events/max(1e-9, frames*HOP/60),
            "detection_delay_p95_ms": float(np.percentile(delays, 95)) if delays else None,
            "seconds": frames*HOP, "threshold": threshold}


def prepare(args):
    import soundfile as sf

    torch.set_num_threads(2)
    args.cache.mkdir(parents=True, exist_ok=True)
    rows, sources = [], {s: set() for s in ("train", "validation", "test")}
    manifests = {}
    for corpus in ("bargein_train", "ami_overlap"):
        manifest = ROOT / corpus / "manifest.jsonl"
        manifests[str(manifest)] = digest(manifest)
        for line in manifest.read_text().splitlines():
            r = json.loads(line)
            # ES2002a/b/c/d share a participant group: hold out the whole group.
            identities = (["ami:" + r["meeting"][:-1]] if corpus == "ami_overlap"
                          else ["libri:" + str(s) for s in r["speakers"]])
            # Prefixes distinguish corpora for leakage audit, not fold assignment.
            split = split_for([s.split(":", 1)[1] for s in identities])
            if split is None:
                continue
            source = Path(r["audio"])
            if not source.is_file():
                raise FileNotFoundError(source)
            target = args.cache / f"{corpus}-{r['id']}.npz"
            if not target.exists():
                audio, rate = sf.read(source, dtype="float32")
                if rate != 16000 or audio.ndim != 1:
                    raise ValueError(f"bad waveform {source}")
                features = logmel(audio)
                labels = np.asarray(r["overlap"], np.float32)
                n = min(len(features), len(labels))
                if abs(len(features)-len(labels)) > 2 or n == 0:
                    raise ValueError(f"label alignment mismatch {source}")
                np.savez(target, features=features[:n].astype(np.float16),
                         labels=labels[:n])
            sources[split].update(identities)
            rows.append({"id": corpus+":"+r["id"], "corpus": corpus,
                         "source_ids": identities, "split": split,
                         "archive": str(target), "audio": str(source)})
            if len(rows) % 250 == 0:
                print(f"prepared {len(rows)} raw-audio examples", flush=True)
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not sources[a] & sources[b], (a, b)
    manifest = args.cache / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r)+"\n" for r in rows))
    report = {"kind": "raw_audio_overlap_data", "command": [sys.executable, *sys.argv],
              "input_manifests": manifests, "manifest_sha256": digest(manifest),
              "split_policy": "Libri speaker and AMI participant-session-group disjoint",
              "source_overlap": 0,
              "counts": {s: sum(r["split"] == s for r in rows) for s in sources}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["counts"]), flush=True)


def load_rows(cache):
    data = {}
    for line in (cache / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        with np.load(row["archive"]) as z:
            x = np.asarray(z["features"], np.float32)
            y = np.asarray(z["labels"], np.float32)
        data.setdefault(row["split"], []).append((row, x, y))
    return data


@torch.no_grad()
def predict(model, rows, device):
    model.eval()
    return [(torch.sigmoid(model(torch.from_numpy(x[None]).to(device)))[0].cpu().numpy(), y)
            for _, x, y in rows]


def train(args):
    from training.context_distill import refuse_gpu_contention

    refuse_gpu_contention()
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    data = load_rows(args.cache)
    model = OverlapNet(args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    history, best, selected = [], -1, None
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    for epoch in range(1, args.epochs+1):
        model.train(); losses = []
        order = rng.permutation(len(data["train"]))
        for offset in range(0, len(order), args.batch):
            batch = [data["train"][i] for i in order[offset:offset+args.batch]]
            longest = max(len(y) for _, _, y in batch)
            x = np.zeros((len(batch), longest, 80), np.float32)
            y = np.zeros((len(batch), longest), np.float32); mask = y.copy()
            for i, (_, feature, label) in enumerate(batch):
                x[i, :len(label)] = feature; y[i, :len(label)] = label
                mask[i, :len(label)] = 1
            logits = model(torch.from_numpy(x).to(device))
            valid = torch.from_numpy(mask).to(device)
            loss = F.binary_cross_entropy_with_logits(
                logits, torch.from_numpy(y).to(device), reduction="none")
            loss = (loss*valid).sum()/valid.sum()
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(loss.item())
        val = predict(model, data["validation"], device)
        scores = [metrics(val, float(t)) for t in np.linspace(.1, .9, 33)]
        score = max(scores, key=lambda s: s["f1"])
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "validation": score, "elapsed_s": time.monotonic()-start}
        history.append(record)
        print(json.dumps(record), flush=True)
        if score["f1"] > best:
            best = score["f1"]; selected = record
            torch.save({"model": model.cpu().state_dict(), "width": args.width,
                        "architecture": "raw_logmel_causal_tcn_v1",
                        "threshold": score["threshold"]}, args.checkpoint)
            model.to(device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    test = {}
    for corpus in ("bargein_train", "ami_overlap"):
        rows = [r for r in data["test"] if r[0]["corpus"] == corpus]
        test[corpus] = metrics(predict(model, rows, device), saved["threshold"])
    # Predeclared bar: frame F1 plus event reliability, all on untouched test.
    passed = all(s["f1"] >= .81 and s["event_recall"] >= .9
                 and s["false_alarms_per_minute"] <= 1.0
                 and s["detection_delay_p95_ms"] is not None
                 and s["detection_delay_p95_ms"] <= 300 for s in test.values())
    report = {"command": [sys.executable, *sys.argv], "history": history,
              "selected": selected, "test": test, "promotion_pass": passed,
              "checkpoint": str(args.checkpoint), "checkpoint_sha256": digest(args.checkpoint),
              "parameters": sum(p.numel() for p in model.parameters()),
              "manifest_sha256": digest(args.cache/"manifest.jsonl"),
              "code_sha256": digest(__file__), "device": str(device),
              "gates": {"f1_min": .81, "event_recall_min": .9,
                        "false_alarms_per_minute_max": 1.0, "delay_p95_ms_max": 300},
              "runtime_policy_changed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"test": test, "promotion_pass": passed}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "train"])
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path,
                   default=REPO/"artifacts/parallel-overlap-tcn-v1.pt")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", choices=["cpu", "mps"], default="mps")
    args = p.parse_args()
    (prepare if args.action == "prepare" else train)(args)


if __name__ == "__main__":
    main()
