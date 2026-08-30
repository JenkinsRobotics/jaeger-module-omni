#!/usr/bin/env python3
"""Train the causal acoustic controller from factual waveform/frame targets.

Each JSONL row points to a source-disjoint NPZ archive containing:

  microphone [samples]        room microphone mixture
  reference [samples]         exact Jaeger playback waveform (zero if absent)
  target [samples]            same room signal with only Jaeger's echo removed
  scene [frames]              0 silence, 1 environment, 2 self, 3 speech
  events [frames, 5]          external, self, music, noise, overlap
  turn [frames]               0 forming, 1 acoustically committed
  wake [frames]               0/1, or -1 where no factual wake label exists

The objective never consumes a transcript, Gemma output, or teacher-forced
answer. Semantic completion and addressee training happen downstream.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from runtime import (  # noqa: E402
    ARCHITECTURE,
    EVENTS,
    HOP_SAMPLES,
    SAMPLE_RATE,
    SCENES,
    TURN_STATES,
    CausalAcousticController,
)


def read_manifest(path: Path) -> dict[str, list[dict]]:
    root = path.expanduser().resolve().parent
    splits = {"train": [], "validation": [], "test": []}
    sources = {name: set() for name in splits}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        split = str(row.get("split", ""))
        if split not in splits:
            raise ValueError(f"{path}:{number}: invalid split {split!r}")
        archive = Path(str(row.get("archive", ""))).expanduser()
        archive = archive if archive.is_absolute() else root / archive
        if not archive.is_file():
            raise FileNotFoundError(archive)
        ids = {str(value) for value in row.get("source_ids", [])}
        if not ids:
            raise ValueError(f"{path}:{number}: source_ids required")
        row["archive"] = archive
        splits[split].append(row)
        sources[split].update(ids)
    for left, right in (("train", "validation"), ("train", "test"),
                        ("validation", "test")):
        overlap = sources[left] & sources[right]
        if overlap:
            raise ValueError(
                f"source leakage {left}/{right}: {sorted(overlap)[:4]}")
    if not all(splits.values()):
        raise ValueError("manifest requires non-empty train/validation/test")
    return splits


def load_example(row: dict, device: torch.device) -> dict[str, torch.Tensor]:
    with np.load(row["archive"], allow_pickle=False) as archive:
        required = {"microphone", "reference", "target", "scene", "events",
                    "turn", "wake"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{row['archive']}: missing {sorted(missing)}")
        arrays = {name: archive[name] for name in required}
    mic = np.asarray(arrays["microphone"], np.float32).reshape(-1)
    reference = np.asarray(arrays["reference"], np.float32).reshape(-1)
    target = np.asarray(arrays["target"], np.float32).reshape(-1)
    if mic.shape != reference.shape or mic.shape != target.shape:
        raise ValueError(f"{row['archive']}: waveform lengths differ")
    frames = len(mic) // HOP_SAMPLES
    if not frames:
        raise ValueError(f"{row['archive']}: empty controller example")
    samples = frames * HOP_SAMPLES
    scene = np.asarray(arrays["scene"], np.int64).reshape(-1)[:frames]
    events = np.asarray(arrays["events"], np.float32)[:frames]
    turn = np.asarray(arrays["turn"], np.int64).reshape(-1)[:frames]
    wake = np.asarray(arrays["wake"], np.float32).reshape(-1)[:frames]
    if (scene.shape != (frames,) or events.shape != (frames, len(EVENTS)) or
            turn.shape != (frames,) or wake.shape != (frames,)):
        raise ValueError(f"{row['archive']}: frame-label shape mismatch")
    if not ((scene >= 0).all() and (scene < len(SCENES)).all() and
            (turn >= 0).all() and (turn < len(TURN_STATES)).all()):
        raise ValueError(f"{row['archive']}: categorical label out of range")

    def waveform(value):
        return torch.from_numpy(value[:samples].reshape(
            1, frames, HOP_SAMPLES)).to(device)

    return {
        "microphone": waveform(mic),
        "reference": waveform(reference),
        "target": waveform(target),
        "scene": torch.from_numpy(scene).to(device),
        "events": torch.from_numpy(events).to(device),
        "turn": torch.from_numpy(turn).to(device),
        "wake": torch.from_numpy(wake).to(device),
    }


def controller_loss(model, batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output = model(batch["microphone"], batch["reference"])
    cleaned = output["cleaned"]
    waveform = functional.smooth_l1_loss(cleaned, batch["target"])
    # Spectral loss stabilizes echo removal without imposing the microphone's
    # phase as a hard target. Frames are flattened back into continuous audio.
    clean_wave = cleaned.reshape(-1)
    target_wave = batch["target"].reshape(-1)
    window = torch.hann_window(320, device=cleaned.device)
    clean_spectrum = torch.stft(
        clean_wave, 512, hop_length=160, win_length=320, window=window,
        return_complex=True)
    target_spectrum = torch.stft(
        target_wave, 512, hop_length=160, win_length=320, window=window,
        return_complex=True)
    spectral = functional.l1_loss(
        torch.log1p(torch.abs(clean_spectrum)),
        torch.log1p(torch.abs(target_spectrum)))
    scene = functional.cross_entropy(
        output["scene_logits"][0], batch["scene"])
    events = functional.binary_cross_entropy_with_logits(
        output["event_logits"][0], batch["events"])
    turn = functional.cross_entropy(
        output["turn_logits"][0], batch["turn"])
    known_wake = batch["wake"] >= 0
    wake = (functional.binary_cross_entropy_with_logits(
        output["wake_logits"][0][known_wake], batch["wake"][known_wake])
        if torch.any(known_wake) else cleaned.new_zeros(()))
    total = (2.0 * waveform + spectral + 0.5 * scene + events +
             0.7 * turn + 0.5 * wake)
    return total, {
        "waveform": waveform, "spectral": spectral, "scene": scene,
        "events": events, "turn": turn, "wake": wake,
    }


def evaluate(model, rows: list[dict], device: torch.device,
             limit: int = 0) -> dict:
    totals = {"samples": 0, "echo_energy": 0.0, "residual_energy": 0.0,
              "near_energy": 0.0, "near_error_energy": 0.0,
              "scene_correct": 0, "scene_frames": 0,
              "turn_correct": 0, "turn_frames": 0,
              "event_tp": np.zeros(len(EVENTS), np.int64),
              "event_fp": np.zeros(len(EVENTS), np.int64),
              "event_fn": np.zeros(len(EVENTS), np.int64)}
    model.eval()
    with torch.inference_mode():
        for row in rows[:limit or None]:
            batch = load_example(row, device)
            output = model(batch["microphone"], batch["reference"])
            clean = output["cleaned"]
            target = batch["target"]
            echo = batch["microphone"] - target
            residual = clean - target
            totals["samples"] += target.numel()
            totals["echo_energy"] += float(torch.sum(echo.square()))
            totals["residual_energy"] += float(torch.sum(residual.square()))
            totals["near_energy"] += float(torch.sum(target.square()))
            totals["near_error_energy"] += float(torch.sum((clean-target).square()))
            scene = output["scene_logits"][0].argmax(-1)
            turn = output["turn_logits"][0].argmax(-1)
            totals["scene_correct"] += int(torch.sum(scene == batch["scene"]))
            totals["scene_frames"] += scene.numel()
            totals["turn_correct"] += int(torch.sum(turn == batch["turn"]))
            totals["turn_frames"] += turn.numel()
            guess = (torch.sigmoid(output["event_logits"][0]) >= 0.5).cpu().numpy()
            truth = (batch["events"] >= 0.5).cpu().numpy()
            totals["event_tp"] += np.sum(guess & truth, axis=0)
            totals["event_fp"] += np.sum(guess & ~truth, axis=0)
            totals["event_fn"] += np.sum(~guess & truth, axis=0)
    eps = 1e-12
    event_f1 = {}
    for index, name in enumerate(EVENTS):
        tp = int(totals["event_tp"][index])
        fp = int(totals["event_fp"][index])
        fn = int(totals["event_fn"][index])
        event_f1[name] = 2 * tp / max(1, 2 * tp + fp + fn)
    return {
        "examples": min(len(rows), limit or len(rows)),
        "echo_reduction_db": 10 * np.log10(
            (totals["echo_energy"] + eps) /
            (totals["residual_energy"] + eps)),
        "target_snr_db": 10 * np.log10(
            (totals["near_energy"] + eps) /
            (totals["near_error_energy"] + eps)),
        "scene_accuracy": totals["scene_correct"] / max(1, totals["scene_frames"]),
        "turn_accuracy": totals["turn_correct"] / max(1, totals["turn_frames"]),
        "event_f1": event_f1,
    }


def atomic_checkpoint(path: Path, model, *, step: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "architecture": ARCHITECTURE,
        "sample_rate": SAMPLE_RATE,
        "hop_samples": HOP_SAMPLES,
        "step": int(step),
        "metrics": metrics,
        "state_dict": {name: value.detach().cpu()
                       for name, value in model.state_dict().items()},
    }, temporary)
    os.replace(temporary, path)


def dry_batch(device: torch.device, frames: int = 40) -> dict:
    rng = np.random.default_rng(0)
    target = rng.normal(0, 0.02, (1, frames, HOP_SAMPLES)).astype(np.float32)
    reference = rng.normal(0, 0.03, target.shape).astype(np.float32)
    microphone = np.clip(target + 0.4 * reference, -1, 1)
    return {
        "microphone": torch.from_numpy(microphone).to(device),
        "reference": torch.from_numpy(reference).to(device),
        "target": torch.from_numpy(target).to(device),
        "scene": torch.full((frames,), 3, dtype=torch.long, device=device),
        "events": torch.tensor(
            [[1, 1, 0, 0, 1]] * frames, dtype=torch.float32, device=device),
        "turn": torch.tensor(
            [0] * (frames - 3) + [1] * 3, dtype=torch.long, device=device),
        "wake": torch.full((frames,), -1.0, device=device),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("training/runs/acoustic-controller.pt"))
    parser.add_argument("--report", type=Path,
                        default=Path("training/runs/acoustic-controller.json"))
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"),
                        default="auto")
    parser.add_argument("--dry-run", action="store_true",
                        help="CPU one-step graph/gradient check; writes nothing")
    args = parser.parse_args()
    if args.dry_run:
        device = torch.device("cpu")
        model = CausalAcousticController().to(device)
        loss, parts = controller_loss(model, dry_batch(device))
        loss.backward()
        finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                     for parameter in model.parameters())
        if not finite:
            raise RuntimeError("non-finite dry-run gradient")
        print(f"acoustic controller dry-run: {model.parameter_count:,} params · "
              f"loss={float(loss.detach()):.4f} · " + " ".join(
                  f"{name}={float(value.detach()):.3f}"
                  for name, value in parts.items()))
        return 0
    if args.manifest is None:
        parser.error("--manifest is required unless --dry-run")
    if args.steps < 1 or args.eval_every < 1:
        parser.error("--steps and --eval-every must be positive")
    device_name = args.device
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    if device_name == "mps":
        from training.context_distill import refuse_gpu_contention
        refuse_gpu_contention()
    device = torch.device(device_name)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    splits = read_manifest(args.manifest)
    model = CausalAcousticController().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(args.seed)
    history = []
    started = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        batch = load_example(rng.choice(splits["train"]), device)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = controller_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(f"step {step}/{args.steps} loss={float(loss):.4f} "
                  f"{(time.time()-started)/step:.3f}s/step", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model, splits["validation"], device, args.eval_examples)
            history.append({"step": step, **metrics})
            print(f"validation step={step} echo={metrics['echo_reduction_db']:.2f}dB "
                  f"scene={metrics['scene_accuracy']:.3f} "
                  f"turn={metrics['turn_accuracy']:.3f}", flush=True)
            snapshot = args.output.with_name(
                f"{args.output.stem}.step{step:06d}{args.output.suffix}")
            atomic_checkpoint(snapshot, model, step=step, metrics=metrics)
    test = evaluate(model, splits["test"], device, 0)
    atomic_checkpoint(args.output, model, step=args.steps, metrics=test)
    report = {
        "kind": "neural_acoustic_controller_training",
        "architecture": ARCHITECTURE,
        "parameters": model.parameter_count,
        "sample_rate": SAMPLE_RATE,
        "hop_samples": HOP_SAMPLES,
        "device": device_name,
        "steps": args.steps,
        "learning_rate": args.lr,
        "seed": args.seed,
        "factual_targets_only": True,
        "transcript_or_gemma_teacher": False,
        "source_disjoint": True,
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "validation": history,
        "test": test,
        "checkpoint": str(args.output.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"test {json.dumps(test, sort_keys=True)}")
    print(f"checkpoint {args.output}\nreport {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
