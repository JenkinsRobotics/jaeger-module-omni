"""Cache the frozen front end across MANY corpora, not just LibriSpeech clean.

LibriSpeech clean is read audiobooks in one acoustic style. A robot hears
spontaneous, accented, noisy speech at varying distance, and an adapter
trained on 12 hours of one style learns that style.

Reuses the parallel agent's AudioFrontEnd and cache layout exactly --
AvgPool5(ReLU(linear1(APM(audio)))), float16 .npy plus manifest.jsonl -- so
everything already built reads these caches unchanged.

    python training/precompute_multi.py --corpus voxpopuli --max-examples 4000
    python training/precompute_multi.py --list

Licences are recorded per corpus because some of what is on the Hub is not
usable in a product: TED-LIUM is CC BY-NC-ND, and NC means reference only.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

SRC = (Path.home() / "GITHUB/VoiceLLM-Playground/Omni Model/gemma-backbone"
       / "train_audio_adapter.py")
CACHE = Path.home() / "models/gemma-audio-adapter/cache"

# name -> (hf_id, config, split, text_field, licence, note)
CORPORA = {
    "librispeech_other": ("openslr/librispeech_asr", "other", "train.500",
                          "text", "CC BY 4.0",
                          "noisier read speech; the hard half of LibriSpeech"),
    "voxpopuli": ("facebook/voxpopuli", "en", "train",
                  "normalized_text", "CC0",
                  "parliamentary, SPONTANEOUS and accented — closest to a "
                  "robot's real input"),
    "fleurs": ("google/fleurs", "en_us", "train",
               "transcription", "CC BY 4.0",
               "clean, many speakers, short utterances"),
    "commonvoice": ("mozilla-foundation/common_voice_17_0", "en", "train",
                    "sentence", "CC0",
                    "huge speaker diversity; needs a HF token + accepted terms"),
    "ami": ("edinburghcstr/ami", "ihm", "train",
            "text", "CC BY 4.0",
            "meeting speech, overlapping talkers, far-field"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", choices=list(CORPORA))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--max-examples", type=int, default=4000)
    ap.add_argument("--min-seconds", type=float, default=0.6)
    ap.add_argument("--max-seconds", type=float, default=10.0)
    ap.add_argument("--cache-dir", type=Path, default=CACHE)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--log-every", type=int, default=250)
    a = ap.parse_args()

    if a.list or not a.corpus:
        print(f"{'name':20s} {'licence':12s} note")
        for k, (_, _, _, _, lic, note) in CORPORA.items():
            print(f"{k:20s} {lic:12s} {note}")
        return 0

    import numpy as np
    from datasets import Audio, load_dataset

    spec = importlib.util.spec_from_file_location("src", SRC)
    src = importlib.util.module_from_spec(spec); spec.loader.exec_module(src)

    hf_id, cfg, split, text_field, lic, note = CORPORA[a.corpus]
    out = a.cache_dir / a.corpus
    (out / "features").mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    done = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    print(f"{a.corpus}: {hf_id} [{cfg}] {split} — {lic}\n  {note}")
    print(f"  {len(done)} already cached", flush=True)

    ds = load_dataset(hf_id, cfg, split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    front = src.AudioFrontEnd(a.device)

    written = seen = 0
    t0 = time.time()
    with manifest.open("a", buffering=1) as f:
        for row in ds:
            if written >= a.max_examples:
                break
            seen += 1
            item = str(row.get("id") or row.get("audio_id")
                       or row.get("path") or seen)
            if item in done:
                continue
            text = str(row.get(text_field) or row.get("text") or "").strip()
            if not text:
                continue
            try:
                wav, rate = src._read_audio(row["audio"])
            except Exception:
                continue
            secs = len(wav) / rate
            if not (a.min_seconds <= secs <= a.max_seconds):
                continue
            feats = front(wav)
            name = src._safe_id(item) + ".npy"
            np.save(out / "features" / name, feats.astype(np.float16))
            f.write(json.dumps({"id": item, "text": text, "seconds": secs,
                                "frames": int(feats.shape[0]),
                                "feature": f"features/{name}",
                                "split": a.corpus, "licence": lic}) + "\n")
            written += 1
            if written % a.log_every == 0:
                el = time.time() - t0
                print(f"  {written} cached / {seen} seen  "
                      f"{el/written:.2f}s/example", flush=True)
    print(f"done: {written} new from {a.corpus} in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
