"""Five-minute room observer: direct audio transcripts plus meeting minutes.

The transcript is an OUTPUT of Gemma, not an ASR handoff into Gemma::

    room PCM -> tuned Whisper encoder -> Jaeger adapter -> Gemma -> transcript

The observer cuts a continuous room into overlapping neural windows because
the trained alignment is accurate on short turns and was measured to collapse
a 44.9-second one-shot decode to one clause.  A 500 ms overlap keeps words at
hard boundaries audible in at least one window.  The complete terminal
transcript may optionally be passed to the already-loaded Gemma only AFTER
audio decoding to create meeting minutes; it is never fed back into the audio
session or used to answer an audio turn.

Examples::

    python -m jaeger_agent_omni.meeting --seconds 300 --out meeting.md
    python -m jaeger_agent_omni.meeting --wav room.wav --k 3 --out meeting.md
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

from jaeger_agent_omni.omni import AUDIO_GGUF, DELTA
from jaeger_agent_omni.runtime.tuned_whisper import (
    DEFAULT_MODEL, TunedWhisperGemmaSession)

RATE = 16_000
WINDOW_S = 9.0
OVERLAP_S = 0.5
DEFAULT_SILERO = (Path.home() / "models/gemma-audio-adapter"
                  / "silero_vad_16k_op15.onnx")


class SpeechAdmission:
    """CPU Silero admission: isolated cable/noise hits cannot open a window."""

    def __init__(self, model_path: str | Path = DEFAULT_SILERO,
                 probability: float = 0.30) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options,
            providers=["CPUExecutionProvider"])
        self.probability = probability

    def voiced_fraction(self, audio: np.ndarray) -> float:
        state = np.zeros((2, 1, 128), dtype=np.float32)
        voiced = total = 0
        frame = 480
        for start in range(0, len(audio) - frame + 1, frame):
            probability, state = self.session.run(None, {
                "input": audio[start:start + frame][None],
                "state": state,
                "sr": np.array(RATE, dtype=np.int64),
            })
            voiced += float(probability[0, 0]) >= self.probability
            total += 1
        return voiced / max(1, total)


def capture(seconds: float, device=None) -> np.ndarray:
    """Record exact float32 mono PCM without running a model in the callback."""
    import sounddevice as sd

    frames = int(round(seconds * RATE))
    print(f"recording {seconds:.0f}s from "
          f"{sd.query_devices(device, 'input')['name']}…", flush=True)
    audio = sd.rec(frames, samplerate=RATE, channels=1, dtype="float32",
                   device=device)
    sd.wait()
    return np.asarray(audio[:, 0], dtype=np.float32)


def read_audio(path: str | Path) -> np.ndarray:
    """Read and, when necessary, resample a mono meeting recording to 16 kHz."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != RATE:
        import scipy.signal
        count = round(len(audio) * RATE / sr)
        audio = scipy.signal.resample(audio, count).astype(np.float32)
    return np.asarray(audio, dtype=np.float32)


def windows(audio: np.ndarray, window_s: float = WINDOW_S,
            overlap_s: float = OVERLAP_S):
    """Yield gap-free, bounded neural windows with explicit time spans."""
    window = int(round(window_s * RATE))
    overlap = int(round(overlap_s * RATE))
    if window <= overlap or overlap < 0:
        raise ValueError("window must be longer than its non-negative overlap")
    hop = window - overlap
    for start in range(0, len(audio), hop):
        end = min(start + window, len(audio))
        segment = audio[start:end]
        if len(segment) < int(0.4 * RATE):
            break
        yield start / RATE, end / RATE, segment
        if end == len(audio):
            break


def minutes_from_transcript(session, transcript: str,
                            max_tokens: int = 512) -> str:
    """Free-run Gemma over its terminal transcript to produce minutes."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_logits_processors

    from jaeger_agent_omni.runtime.tuned_whisper import (
        _token_loop_blocker, _word_loop_blocker)

    messages = [
        {"role": "system", "content": (
            "Turn the speaker-attributed transcript into concise meeting "
            "minutes. Include topics, decisions, action items with owners "
            "when stated, and open questions. Do not invent missing facts; "
            "treat [inaudible] and overlapping time ranges as uncertainty.")},
        {"role": "user", "content": transcript},
    ]
    prompt = session.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = session.tokenizer.encode(prompt, add_special_tokens=False)
    ids = mx.array(prompt_ids, dtype=mx.int32)
    blocked = {
        int(session.tokenizer.convert_tokens_to_ids("<|think|>")),
        int(session.tokenizer.convert_tokens_to_ids("<|channel>")),
    }
    processors = make_logits_processors(
        logit_bias={token: -1e9 for token in blocked},
        repetition_penalty=1.05,
        repetition_context_size=20,
    )
    processors.extend((
        _token_loop_blocker(4, 1),
        _word_loop_blocker(session.tokenizer, 4, 1),
    ))
    generated = []
    for token, _ in generate_step(
            ids, session.model, max_tokens=max_tokens,
            logits_processors=processors):
        token = int(token)
        if token in (1, 106):
            break
        generated.append(token)
    return session.tokenizer.decode(
        generated, skip_special_tokens=True).strip()


def observe(audio: np.ndarray, adapter: str | Path = DELTA,
            model: str | Path = DEFAULT_MODEL,
            audio_gguf: str | Path = AUDIO_GGUF, k: int = 2,
            window_s: float = WINDOW_S,
            overlap_s: float = OVERLAP_S,
            make_minutes: bool = True,
            min_speech_fraction: float = 0.25,
            silero_model: str | Path = DEFAULT_SILERO,
            front_end: str = "tuned",
            whisper: str = "openai/whisper-medium") \
            -> tuple[list[dict], str | None]:
    """Decode every window free-running, then optionally write final minutes."""
    from jaeger_agent_omni.runtime.speaker import OnlineSpeakerTracker

    tracker = OnlineSpeakerTracker(k)
    admission = SpeechAdmission(silero_model)
    session = None
    rows = []
    for index, (start, end, segment) in enumerate(
            windows(audio, window_s, overlap_s)):
        speech_fraction = admission.voiced_fraction(segment)
        admitted = speech_fraction >= min_speech_fraction
        label = (tracker.identify(segment) or "unknown") if admitted \
            else "unknown"
        if not admitted:
            row = {"window": index, "start_s": start, "end_s": end,
                   "speaker": label, "text": "[no speech admitted]",
                   "ms": 0.0, "speech_fraction": speech_fraction,
                   "admitted": False}
            rows.append(row)
            print(f"[{start:6.1f}-{end:6.1f}] [speaker unknown] "
                  f"[no speech admitted] (Silero {speech_fraction:.1%})",
                  flush=True)
            continue
        if session is None:
            if front_end == "stock":
                from jaeger_agent_omni.runtime.stock_whisper import (
                    StockWhisperGemmaSession)
                session = StockWhisperGemmaSession(
                    adapter, model_path=model, whisper_id=whisper,
                    system=None)
            elif front_end == "tuned":
                session = TunedWhisperGemmaSession(
                    adapter, model_path=model, audio_gguf=audio_gguf,
                    system=None)
            else:
                raise ValueError(f"unknown meeting front end {front_end!r}")
        session.reset()
        session.open_turn()
        for offset in range(0, len(segment), 1600):
            session.feed_audio(segment[offset:offset + 1600])
        started = time.monotonic()
        session.end_of_turn()
        text = session.respond(max_tokens=160, eos_ids={1, 106}).strip()
        elapsed_ms = (time.monotonic() - started) * 1000
        if not text:
            text = "[inaudible]"
        row = {"window": index, "start_s": start, "end_s": end,
               "speaker": label, "text": text, "ms": elapsed_ms,
               "speech_fraction": speech_fraction, "admitted": True}
        rows.append(row)
        print(f"[{start:6.1f}-{end:6.1f}] [speaker {label}] "
              f"{text} ({elapsed_ms:.0f} ms)", flush=True)
    transcript = "\n".join(
        f"[{row['start_s']:06.1f}-{row['end_s']:06.1f}] "
        f"[speaker {row['speaker']}] {row['text']}"
        for row in rows if row["admitted"])
    minutes = (minutes_from_transcript(session, transcript)
               if make_minutes and transcript and session is not None else None)
    del admission, tracker
    if session is not None:
        del session
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except (ImportError, AttributeError):
        pass
    return rows, minutes


def render(rows: list[dict], minutes: str | None, duration_s: float,
           window_s: float, overlap_s: float) -> str:
    """Render an auditable transcript: every neural window keeps its timecode."""
    lines = [
        "# Meeting observer",
        "",
        f"Duration: {duration_s:.1f} seconds. Neural windows: {window_s:.1f}s "
        f"with {overlap_s:.1f}s overlap.",
        "",
        "Speaker labels identify the dominant voice near each window end; "
        "they are turn-level telemetry, not word-level diarization.",
        "Windows marked `[no speech admitted]` were rejected by CPU Silero "
        "before neural decoding and cannot contribute hallucinated minutes.",
        "",
        "## Transcript",
        "",
    ]
    lines.extend(
        f"- `{row['start_s']:06.1f}-{row['end_s']:06.1f}` "
        f"**[speaker {row['speaker']}]** {row['text']}"
        for row in rows)
    if minutes:
        lines.extend(("", "## Minutes", "", minutes))
    return "\n".join(lines).strip() + "\n"


def demo() -> None:
    """Check five-minute coverage without loading models or audio devices."""
    audio = np.zeros(300 * RATE, dtype=np.float32)
    spans = list(windows(audio))
    assert spans[0][0] == 0 and spans[-1][1] == 300
    assert len(spans) == 36
    for left, right in zip(spans, spans[1:]):
        assert abs(left[1] - right[0] - OVERLAP_S) < 1e-6
    page = render([{"start_s": 0.0, "end_s": 9.0, "speaker": "A",
                    "text": "hello", "ms": 1.0, "admitted": True,
                    "speech_fraction": 1.0}], "none", 9, 9, .5)
    assert "[speaker A]" in page and "## Minutes" in page
    print("meeting: five-minute gap-free overlapped windows and artifact OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = ap.add_mutually_exclusive_group(required="--demo" not in sys.argv)
    source.add_argument("--wav")
    source.add_argument("--seconds", type=float)
    ap.add_argument("--device", help="input device id or exact name")
    ap.add_argument("--k", type=int, default=2,
                    help="maximum room voices; use 3 when agent playback leaks")
    ap.add_argument("--window-seconds", type=float, default=WINDOW_S)
    ap.add_argument("--overlap-seconds", type=float, default=OVERLAP_S)
    ap.add_argument("--adapter", default=str(DELTA))
    ap.add_argument("--front-end", choices=("tuned", "stock"),
                    default="tuned")
    ap.add_argument("--whisper", default="openai/whisper-medium",
                    help="stock Whisper checkpoint id")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--audio-gguf", default=str(AUDIO_GGUF))
    ap.add_argument("--silero", default=str(DEFAULT_SILERO))
    ap.add_argument("--min-speech-fraction", type=float, default=0.25,
                    help="fraction of 30ms frames Silero must admit")
    ap.add_argument("--out", default="meeting.md")
    ap.add_argument("--record", help="save exact captured PCM here")
    ap.add_argument("--no-minutes", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        demo()
        return 0
    device = int(args.device) if args.device and args.device.isdigit() \
        else args.device
    audio = (read_audio(args.wav) if args.wav
             else capture(args.seconds, device=device))
    if args.record:
        import soundfile as sf
        sf.write(args.record, audio, RATE)
    rows, minutes = observe(
        audio, adapter=args.adapter, model=args.model,
        audio_gguf=args.audio_gguf, k=args.k,
        window_s=args.window_seconds, overlap_s=args.overlap_seconds,
        make_minutes=not args.no_minutes,
        min_speech_fraction=args.min_speech_fraction,
        silero_model=args.silero, front_end=args.front_end,
        whisper=args.whisper)
    artifact = render(rows, minutes, len(audio) / RATE,
                      args.window_seconds, args.overlap_seconds)
    Path(args.out).write_text(artifact)
    print(f"wrote {args.out}: {len(rows)} windows, "
          f"{len(audio)/RATE:.1f}s retained", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
