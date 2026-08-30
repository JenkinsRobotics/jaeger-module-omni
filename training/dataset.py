"""Capture corpus -> training batches.

Each turn yields three things, and which of them matter depends on the stage:

    waveform    the student's input. Audio only, no transcript.
    teacher_ids the transcript, tokenised. The teacher's input — Stage 4 runs
                the same frozen backbone over these to produce the targets the
                audio path must match.
    labels      what the student should emit.
                  Stage 3 warmup  -> the transcript. Cheap, stable, lands the
                                     projector in roughly the right region.
                  Stage 4 distill -> the response.

Stage 3 predicting the transcript is not the goal — it is scaffolding. If we
stopped there we would have built an ASR with extra steps. Stage 4 is where
the audio path learns to *reach the same conclusion* the text path reaches.

    python training/dataset.py     # self-check, synthesises its own corpus
"""

from __future__ import annotations

import sys
from pathlib import Path

from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jaeger_agent_omni.core.capture import SAMPLE_RATE, CaptureReader  # noqa: E402

WARMUP_TARGET = "transcript"
DISTILL_TARGET = "response"
IGNORE_INDEX = -100

# Concurrent full duplex is the default operating mode, so it is the mode the
# model is trained for. In concurrent operation the projector receives audio
# that is MID-UTTERANCE — 1.2 s of a sentence still being spoken. A projector
# trained only on complete utterances has never seen that shape, and the
# mismatch shows up as degraded streaming rather than as an error.
#
# Prefix augmentation truncates a fraction of clips at a random point. Two
# things fall out of it:
#
#   1. the projector's representation of a partial utterance is in-distribution
#   2. turn-end labels come free — a truncated clip is "still speaking" (0),
#      a complete one is "turn ended" (1) — from any corpus, with no
#      conversational data required
#
# Half duplex ("ignore input while I speak") is a runtime policy on top. The
# model does not change; the organiser chooses.
MIN_PREFIX_S = 0.6      # shorter than this carries too little to supervise


class CaptureDataset(torch.utils.data.Dataset):
    """Turns from a capture corpus, filtered and tokenised.

    Holds paths, not audio — a corpus outgrows RAM long before it outgrows
    disk, and wav decode is cheap next to a forward pass.

    `split` carves out a held-out set. Without one, `target_acc` cannot tell
    learning from memorising, and a small corpus memorises fast: 0.91 on data
    the model has already seen means nothing at all.

    The split is deterministic and by turn id, not by position. Sequential
    slicing on a corpus imported from one recording would put the end of the
    speech in validation and the beginning in training — different topic,
    different delivery, and an unfairly hard eval.
    """

    def __init__(self, root, processor, tokenizer, *, stage: str = "stage3_warmup",
                 min_confidence: float | None = 0.8, max_audio_s: float = 30.0,
                 max_label_tokens: int = 64, split: str = "all",
                 val_fraction: float = 0.1, prefix_fraction: float = 0.0,
                 seed: int = 0):
        if split not in ("all", "train", "val"):
            raise ValueError(f"split must be all/train/val, got {split!r}")
        self.reader = CaptureReader(root, min_confidence=min_confidence)
        self.processor = processor
        self.tokenizer = tokenizer
        self.target = WARMUP_TARGET if stage == "stage3_warmup" else DISTILL_TARGET
        self.max_label_tokens = max_label_tokens
        self.split = split
        self.eos_id = getattr(tokenizer, "eos_token_id", None)
        self.prefix_fraction = prefix_fraction
        self.seed = seed

        turns = [t for t in self.reader if 0 < t.duration_s <= max_audio_s]
        if split != "all":
            # Hash the id: stable across runs, independent of file order, and
            # unaffected by turns being added to the corpus later.
            def is_val(t):
                return (int(t.turn_id, 16) % 1000) < int(val_fraction * 1000)
            turns = [t for t in turns if is_val(t) == (split == "val")]

        self.turns = turns
        if not self.turns:
            raise ValueError(
                f"no usable turns in {root} (split={split}) — check "
                f"min_confidence ({min_confidence}) and max_audio_s ({max_audio_s})")

    def __len__(self) -> int:
        return len(self.turns)

    def _encode(self, text: str, add_eos: bool = False) -> torch.Tensor:
        ids = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=self.max_label_tokens).input_ids[0]
        if add_eos and self.eos_id is not None and (
                len(ids) == 0 or ids[-1].item() != self.eos_id):
            ids = torch.cat([ids, torch.tensor([self.eos_id])])
        return ids

    def _prefix_cut(self, i: int, duration_s: float) -> float | None:
        """Where to truncate this clip, or None to keep it whole.

        Deterministic per index: the same item is cut the same way every epoch,
        so validation is stable and a resumed run does not silently change its
        data distribution.
        """
        if self.prefix_fraction <= 0 or duration_s <= MIN_PREFIX_S * 1.5:
            return None
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + i)
        if torch.rand(1, generator=g).item() >= self.prefix_fraction:
            return None
        lo, hi = MIN_PREFIX_S, duration_s * 0.9
        return lo + (hi - lo) * torch.rand(1, generator=g).item()

    def __getitem__(self, i: int) -> dict:
        turn = self.turns[i]
        audio = self.reader.audio(turn)
        duration = turn.duration_s

        cut = self._prefix_cut(i, duration)
        is_complete = cut is None
        if not is_complete:
            audio = audio[: int(cut * SAMPLE_RATE)]
            duration = len(audio) / SAMPLE_RATE

        feats = self.processor(audio, sampling_rate=SAMPLE_RATE,
                               return_tensors="pt").input_features[0]

        # A truncated clip's transcript no longer matches its audio, so it
        # cannot supervise transcription. It still supervises turn-end, which
        # is the point. Labels are masked out entirely for prefixes.
        target = getattr(turn, self.target)
        labels = (self._encode(target, add_eos=True) if is_complete
                  else torch.tensor([IGNORE_INDEX], dtype=torch.long))

        return {
            "features": feats,
            "duration_s": duration,
            "teacher_ids": self._encode(turn.transcript),
            # EOS on the target only. Without it the model is never taught to
            # stop, so at generation time it runs to max_new_tokens and
            # confabulates past the end of the utterance — which shows up as a
            # WER far above 100% from insertions alone.
            "labels": labels,
            # Free supervision for the Stage 5 turn-end head.
            "turn_ended": float(is_complete),
        }


def collate(batch: list, pad_id: int) -> dict:
    """Pad to the batch max. Label padding is IGNORE_INDEX, not pad_id —
    otherwise the loss rewards predicting padding, which it will happily do."""
    def pad(key, value):
        n = max(len(b[key]) for b in batch)
        out = torch.full((len(batch), n), value, dtype=torch.long)
        for i, b in enumerate(batch):
            out[i, : len(b[key])] = b[key]
        return out

    return {
        "features": torch.stack([b["features"] for b in batch]),
        "durations": [b["duration_s"] for b in batch],
        "teacher_ids": pad("teacher_ids", pad_id),
        "labels": pad("labels", IGNORE_INDEX),
        "turn_ended": torch.tensor([b.get("turn_ended", 1.0) for b in batch]),
    }


def loader(dataset, batch_size: int, pad_id: int, shuffle: bool = True):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=lambda b: collate(b, pad_id))


def demo():
    import tempfile

    from jaeger_agent_omni.core.capture import CaptureWriter

    class FakeProcessor:
        def __call__(self, audio, sampling_rate, return_tensors):
            # Stand in for whisper's 80x3000 mel; only the shape matters here.
            return type("R", (), {"input_features": torch.zeros(1, 80, 3000)})()

    class FakeTok:
        pad_token_id = 0
        eos_token_id = 99

        def __call__(self, text, return_tensors=None, truncation=False,
                     max_length=None):
            ids = torch.tensor([[len(w) for w in text.split()] or [1]])
            if max_length:
                ids = ids[:, :max_length]
            return type("R", (), {"input_ids": ids})()

    with tempfile.TemporaryDirectory() as tmp:
        w = CaptureWriter(tmp)
        tone = np.sin(np.arange(SAMPLE_RATE) * 0.02).astype(np.float32)
        w.write(tone, transcript="what is two plus two", response="Four.",
                confidence=0.95)
        w.write(tone, transcript="hello there friend", response="Hi.",
                confidence=0.91)
        w.write(tone, transcript="garbled nonsense", response="?",
                confidence=0.20)                      # below the gate
        w.write(np.zeros(SAMPLE_RATE * 40, np.float32), transcript="very long",
                response="x", confidence=0.99)        # past max_audio_s

        ds = CaptureDataset(tmp, FakeProcessor(), FakeTok(), min_confidence=0.8)
        # Low-confidence and over-length turns are both excluded.
        assert len(ds) == 2, len(ds)

        item = ds[0]
        assert item["features"].shape == (80, 3000)
        assert item["duration_s"] == 1.0

        # Warmup targets the transcript; distill targets the response. Getting
        # this backwards trains an ASR and looks like it is working.
        warm = CaptureDataset(tmp, FakeProcessor(), FakeTok(), stage="stage3_warmup")
        dist = CaptureDataset(tmp, FakeProcessor(), FakeTok(), stage="stage4_distill")
        assert warm.target == "transcript" and dist.target == "response"
        assert not torch.equal(warm[0]["labels"], dist[0]["labels"])

        # Targets end with EOS; the teacher's input does not. Without this the
        # model never learns where an utterance stops.
        assert item["labels"][-1].item() == FakeTok.eos_token_id
        assert item["teacher_ids"][-1].item() != FakeTok.eos_token_id

        batch = collate([ds[0], ds[1]], pad_id=0)
        assert batch["features"].shape == (2, 80, 3000)
        assert batch["teacher_ids"].shape[0] == 2
        assert len(batch["durations"]) == 2

        # Label padding must be IGNORE_INDEX. Padding with pad_id would teach
        # the model that emitting padding is correct.
        n = batch["labels"].shape[1]
        short = min(len(ds[0]["labels"]), len(ds[1]["labels"]))
        if n > short:
            assert (batch["labels"] == IGNORE_INDEX).any(), "labels not masked"
        assert (batch["teacher_ids"] != IGNORE_INDEX).all(), "teacher must not be masked"

        dl = loader(ds, batch_size=2, pad_id=0, shuffle=False)
        got = next(iter(dl))
        assert got["features"].shape[0] == 2

        # Train and val must partition the corpus: no overlap, nothing lost.
        many = CaptureWriter(tmp)
        for i in range(60):
            many.write(tone, transcript=f"utterance number {i}",
                       response=f"reply {i}", confidence=0.95)

        tr = CaptureDataset(tmp, FakeProcessor(), FakeTok(), split="train",
                            val_fraction=0.2)
        va = CaptureDataset(tmp, FakeProcessor(), FakeTok(), split="val",
                            val_fraction=0.2)
        every = CaptureDataset(tmp, FakeProcessor(), FakeTok(), split="all")

        tr_ids = {t.turn_id for t in tr.turns}
        va_ids = {t.turn_id for t in va.turns}
        assert not (tr_ids & va_ids), "train and val overlap"
        assert tr_ids | va_ids == {t.turn_id for t in every.turns}, "turns lost"
        assert 0 < len(va) < len(tr), (len(va), len(tr))

        # Deterministic: the same corpus must split identically next run, or
        # tonight's validation set is tomorrow's training set.
        again = CaptureDataset(tmp, FakeProcessor(), FakeTok(), split="val",
                               val_fraction=0.2)
        assert {t.turn_id for t in again.turns} == va_ids

        # --- prefix augmentation: training for concurrent full duplex -------
        # Long clips so truncation has room; 2 s at 16 kHz.
        long_tone = np.sin(np.arange(SAMPLE_RATE * 2) * 0.02).astype(np.float32)
        pfx_dir = Path(tmp) / "pfx"
        pw = CaptureWriter(pfx_dir)
        for i in range(40):
            pw.write(long_tone, transcript=f"utterance {i}", response=f"r{i}",
                     confidence=0.95)

        whole = CaptureDataset(pfx_dir, FakeProcessor(), FakeTok(),
                               prefix_fraction=0.0)
        mixed = CaptureDataset(pfx_dir, FakeProcessor(), FakeTok(),
                               prefix_fraction=0.5)

        # With no augmentation every clip is complete and fully supervised.
        assert all(whole[i]["turn_ended"] == 1.0 for i in range(len(whole)))

        ended = [mixed[i]["turn_ended"] for i in range(len(mixed))]
        assert 0.0 in ended and 1.0 in ended, "augmentation produced only one class"
        n_cut = ended.count(0.0)
        assert 8 <= n_cut <= 32, f"{n_cut}/40 cut — fraction is far off 0.5"

        for i in range(len(mixed)):
            item = mixed[i]
            if item["turn_ended"] == 0.0:
                # Truncated: shorter than the original, and NOT supervised for
                # transcription — its transcript no longer matches its audio.
                assert item["duration_s"] < 2.0, item["duration_s"]
                assert item["duration_s"] >= MIN_PREFIX_S * 0.99
                assert (item["labels"] == IGNORE_INDEX).all(), \
                    "a prefix must not supervise transcription"
            else:
                assert abs(item["duration_s"] - 2.0) < 1e-6
                assert (item["labels"] != IGNORE_INDEX).any()

        # Deterministic per index: the same item is cut the same way every
        # epoch, so a resumed run does not silently change its data.
        again2 = CaptureDataset(pfx_dir, FakeProcessor(), FakeTok(),
                                prefix_fraction=0.5)
        assert [again2[i]["turn_ended"] for i in range(len(again2))] == ended
        assert all(abs(again2[i]["duration_s"] - mixed[i]["duration_s"]) < 1e-9
                   for i in range(len(mixed)))

        # A different seed must cut differently, or "augmentation" is one
        # fixed dataset wearing a hat.
        other = CaptureDataset(pfx_dir, FakeProcessor(), FakeTok(),
                               prefix_fraction=0.5, seed=7)
        assert [other[i]["turn_ended"] for i in range(len(other))] != ended

        # turn_ended survives collation — it is the head's label.
        b = collate([mixed[i] for i in range(4)], pad_id=0)
        assert b["turn_ended"].shape == (4,)

        # An empty corpus must say so rather than yield zero batches silently.
        try:
            CaptureDataset(tmp, FakeProcessor(), FakeTok(), min_confidence=0.999)
        except ValueError as exc:
            assert "no usable turns" in str(exc)
        else:
            raise AssertionError("empty corpus must raise")

    print(f"dataset: {len(ds)} turns kept of 4 (confidence + length gates)")
    print(f"dataset: prefix augmentation cut {n_cut}/40 clips, "
          f"turn-end labels free")
    print("dataset: stage targets, padding, ignore-index, determinism OK")


if __name__ == "__main__":
    demo()
