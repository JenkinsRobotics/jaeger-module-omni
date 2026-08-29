#!/usr/bin/env python3
"""Build the fixed named-address audio gate with Kokoro.

Text is used only to synthesize evaluation WAVs.  The resulting companion
gate remains free-running and receives audio plus expected RESPOND labels; no
transcript enters the runtime model path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jaeger_agent_omni.voice import speak_kokoro


QUESTIONS = (
    ("q00", "Hey Jaeger, what is two plus two?", "four"),
    ("q01", "Hey Jaeger, what color is the sky?", "blue"),
    ("q02", "Hey Jaeger, how many days in a week?", "seven"),
    ("q03", "Hey Jaeger, what is the capital of France?", "paris"),
    ("q04", "Hey Jaeger, what is ten minus three?", "seven"),
    ("q05", "Hey Jaeger, how many legs does a spider have?", "eight"),
    ("q06", "Hey Jaeger, what is the largest planet?", "jupiter"),
    ("q07", "Hey Jaeger, what is five times five?", "twenty five"),
    ("q08", "Hey Jaeger, what season comes after summer?", "autumn"),
    ("q09", "Hey Jaeger, how many hours in a day?", "twenty four"),
    ("q10", "Hey Jaeger, what is the freezing point of water?",
     "zero degrees celsius"),
    ("q11", "Hey Jaeger, who wrote Romeo and Juliet?", "shakespeare"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path,
                        default=Path("/tmp/jaeger-companion-addressed"))
    parser.add_argument("--voice", default="af_heart")
    args = parser.parse_args()
    args.dir.mkdir(parents=True, exist_ok=True)
    cases = []
    manifest = []
    for key, transcript, answer in QUESTIONS:
        destination = args.dir / f"{key}.wav"
        speak_kokoro(transcript, destination, args.voice, 1.0)
        cases.append({"key": key, "transcript": transcript,
                      "answers": [answer]})
        manifest.append(json.dumps({
            "audio": str(destination.resolve()), "expected": "respond",
            "category": "named_direct_question"}))
    (args.dir / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    manifest_path = args.dir.parent / "jaeger-companion-addressed.jsonl"
    manifest_path.write_text("\n".join(manifest) + "\n")
    print(f"{len(cases)} addressed WAVs -> {args.dir}")
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
